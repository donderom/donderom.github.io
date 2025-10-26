---
title: "Compositional streaming of services with FS2"
description: "Everything's a stream: building composable FS2 services"
date: 2020-05-23T12:56:19+02:00
draft: false
---

The intention of this post is to demonstrate a blueprint for composing multiple services as streams within the scope of one process with [FS2](https://fs2.io). The idea is that, when dealing with unbounded I/O streams, modeling everything as streams leads to better composition.

The libraries that will be used for all the examples:

```scala
"co.fs2" %% "fs2-core" % "2.3.0",
// Configuration and validation
"is.cir" %% "ciris-refined" % "1.0.4"
```

And the imports:

```scala
import cats.effect._
import cats.implicits._
// Configuration and validation
import ciris._
import ciris.refined._
import eu.timepit.refined.W
import eu.timepit.refined.api.Refined
import eu.timepit.refined.string.MatchesRegex
import eu.timepit.refined.types.net.PortNumber
import eu.timepit.refined.types.string.NonEmptyString
```

## Configuration
Let's start with configuration. Unlike the rest of the examples, I'd like to pause and show the awesomeness of the [Ciris](https://cir.is/) configuration library. The docs cover most of its features, so let's jump to defining a sample config for Elasticsearch. The basic config would probably contain the host and port number, at least. If it's deployed on AWS, it could require the AWS region as well (along with many other parameters skipped for brevity). The region names follow a particular pattern so let's start by validating it first:

```scala
object AWS {
  type Region = String Refined MatchesRegex[W.`"([a-z]{2,}-)+[0-9]"`.T]
}
```

We're using the [refined](https://github.com/fthomas/refined) library here to specify the `Region` type, which checks if a string matches the region name regex.

The full config will look like this:

```scala
final case class ElasticsearchConfig(
  host: NonEmptyString,
  port: PortNumber,
  region: AWS.Region
)
```

Notice how every field is validated using special types from `refined`. The `host` field is a string but cannot be empty, while the `port` number is guaranteed to be in the correct range by using the `PortNumber` type. As a result, there are no `String`s and `Int`s all over the place with custom validation on top. Ok, it's just a model, time to specify how to populate it with values.

```scala
object ElasticsearchConfig {
  val config: ConfigValue[ElasticsearchConfig] =
    (
      env("ELASTICSEARCH_HOST").default("localhost").as[NonEmptyString],
      env("ELASTICSEARCH_PORT").as[PortNumber].default(PortNumber(9200)),
      env("ELASTICSEARCH_REGION").as[AWS.Region]
    ).parMapN(apply)
}
```

The values are loaded from environment variables with defaults, validated, and combined in parallel. The automatic support for `refined` is provided by the  [ciris-refined](https://cir.is/docs/modules#refined) module. The same approach is used to compose multiple configurations into larger ones. For example given a separate Kafka config

```scala
final case class KafkaConfig(bootstrapServers: NonEmptyString, ...)
```

both configs can be accumulated into one main config:

```scala
final case class Config(es: ElasticsearchConfig, kafka: KafkaConfig)

object Config {
  val config: ConfigValue[Config] = (
    ElasticsearchConfig.config,
    KafkaConfig.config
  ).parMapN(apply)
}
```

To load the configuration, we have to specify an effect type, which in this case will be cats-effect's [`IO`](https://typelevel.org/cats-effect/docs/2.x/datatypes/io) type:

```scala
val config: IO[Config] = Config.config.load[IO]
```

## Dependencies
An initial stream to work with can be built by evaluating the `IO` effect to emit the config value (effectively going from `IO[A]` to `Stream[IO, A]`):

```scala
val stream: fs2.Stream[IO, Config] = fs2.Stream.eval(config)
```

We can go back any time with `stream.compile.lastOrError: IO[Config]`. The main reason not to, is to model an effectful computation as a stream, allowing it to be composed with a streaming service later on. In other words, instead of "wrapping" a dependency inside a stream, this modeling approach represents it as a stream itself.

Another dependency is an Elasticsearch client. Here's a complete example of the client interface with a dummy implementation:

```scala
trait Event

trait ElasticsearchClient[F[_]] {
  def index(event: Event): F[Unit]
}

object ElasticsearchClient {
  def create[F[_]](config: ElasticsearchConfig)(
    implicit F: Sync[F]
  ): ElasticsearchClient[F] =
    new ElasticsearchClient[F] {
      def index(event: Event): F[Unit] =
        F.delay(println(s"Indexing $event to ${config.host}:${config.port}"))
    }
}
```

The config is required to create a new client. There is already a stream emitting the config while evaluating the effect. The basic composition should be familiar:

```scala
stream.map(config => ElasticsearchClient.create[IO](config.es))
// fs2.Stream[IO, ElasticsearchClient[IO]] = Stream(..)
```

Although it's completely valid, the idea is to stick with modeling everything as streams. Additionally, the config could be needed downstream, making both the config and the client dependencies.

```scala
val stream: fs2.Stream[IO, ElasticsearchClient[IO]] = for {
  config <- fs2.Stream.eval(Config.config.load[IO])
  client <- fs2.Stream.emit(ElasticsearchClient.create[IO](config.es))
} yield client
```

At this step, configuration loading and client creation emit one value, so they can remain in `IO` without any problem. The composition would be similar. When FS2 shines, though, is in dealing with I/O computations in constant memory.

## Composition
It's common to run both an HTTP server and a Kafka (or any other) consumer concurrently within the scope of one microservice. Continuing this example, an HTTP server could serve data from an Elasticsearch index, while a consumer would index data from a Kafka topic(s). {{< sidenote >}}The code will be more abstract but two good libraries with FS2 support are [http4s](https://http4s.org/) and [fs2-kafka](https://fd4s.github.io/fs2-kafka/) respectively.{{< /sidenote >}}

A hypothetical API for a consumer that requires its own config and an Elasticsearch client producing a stream:

```scala
def eventStream[F[_]](config: KafkaConfig, client: ElasticsearchClient[F])(
  implicit F: ConcurrentEffect[F]
): fs2.Stream[F, Unit] = ???
```

And an HTTP server API that is modeled as cats-effect's [`Resource`](https://typelevel.org/cats-effect/docs/2.x/datatypes/resource):

```scala
trait Server[F[_]]

def server[F[_]](implicit F: ConcurrentEffect[F]): Resource[F, Server[F]] = ???
```

Abstracting over effect type (all these `F[_]`) following the least powerful typeclass rule is crucial for stream composition and testing. There is no need to constrain an effect with `ConcurrentEffect` when `Sync` or `Applicative` would be enough.

Given that the consumer and the server are independent services, they cannot be composed like dependencies, as they are supposed to run *concurrently*. Instead of "sequential" `flatMap`, we can use `parJoin`, which will nondeterministically merge a stream of streams into a single stream. To run multiple streams concurrently with FS2:

```scala {hl_lines=7}
val stream: fs2.Stream[IO, Unit] = for {
  config <- fs2.Stream.eval(Config.config.load[IO])
  client <- fs2.Stream.emit(ElasticsearchClient.create[IO](config.es))
  app <- fs2.Stream(
    eventStream(config.kafka, client),
    fs2.Stream.resource(server[IO]) // Resource[F, Server[F]] → Stream[F, Server[F]]
  ).parJoinUnbounded
} yield app
```

And that's it! More information about `parJoin` and `parJoinUnbounded` {{< sidenote >}}For this example `parJoin(2)` and `parJoinUnbounded` are interchangeable but when adding one more service it's easy to forget to bump the `maxOpen` arguments of `parJoin`.{{< /sidenote >}}, as well as many other composition primitives such as `concurrently` (when the services depend on each other), can be found [here](https://s01.oss.sonatype.org/service/local/repositories/releases/archive/co/fs2/fs2-core_2.12/2.3.0/fs2-core_2.12-2.3.0-javadoc.jar/!/fs2/Stream.html). The complete snippet of the main entry point with basic error handling would look like this:

```scala
object Main extends IOApp {
  def run(args: List[String]): IO[ExitCode] = {
    val stream: fs2.Stream[IO, Unit] = for {
      config <- fs2.Stream.eval(Config.config.load[IO])
      client <- fs2.Stream.emit(ElasticsearchClient.create[IO](config.es))
      app <- fs2.Stream(
        eventStream(config.kafka, client),
        fs2.Stream.resource(server[IO])
      ).parJoinUnbounded
    } yield app

    stream
      .compile
      .drain
      .onError {
        case e: Throwable => IO(println(e.getMessage))
      }
      .as(ExitCode.Success)
  }
  // ...
}
```

Not only is the flow concise, but every line packs a punch. The part with `eval` includes type-safe configuration loading semantics, performs it in parallel with validation, and evaluates an effect. Then, any value can be represented as a stream, be it effectful (with `eval`) or not (with `emit`, where a sequence of values is handled with `emits`). The main part comprises an unbounded stream of messages and a server with built-in resource safety (`resource`) that run concurrently (`parJoin`, where service independence can be controlled by changing the concurrency operation). Adding a third service to the example (let say a message producer along with the consumer) won't change the structure at all. To finish things off, there is a centralized place for error handling and straightforward support for testing. Mocking every dependency and service in unit tests can be avoided by replacing the `IO` effect with another. Streaming semantics are already tested by the FS2 library itself, which allows focusing on service logic.

> *Why not use compositional streaming modeling for applications with streaming core?*
