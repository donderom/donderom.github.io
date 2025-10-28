---
title: "Scaling effects in Scala"
description: "Horizontal and vertical effect scaling in Scala"
date: 2019-02-10T15:32:58+02:00
draft: false
features: ["diag"]
---

42\. A value. Different from 37 but both are numbers.

***Effect* is to type what type is to value.**

🄰s type allows to abstract over value, effect allows to abstract over type.

```scala
42: Int
"text": String
```

```scala
def opt[A](value: A): Option[A]
opt(42): Option[Int]
opt("text"): Option[String]
```

There are two types of effect scaling---horizontal ⇄ and vertical ⇅.

## Horizontal Scaling

The simplest scaling approach that replaces an effect with another one.

Built-in effects: `Option[A] → Future[A]`.

Third-party effects: `Future[A] → Task[A]`. {{< marginnote >}}https://monix.io{{< /marginnote >}}

```scala
import scala.concurrent.Future
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global

val future = Future(42)
val task = Task.deferFuture(future) // Future[A] → Task[A]
task.runToFuture: Future[Int]       // Task[A] → Future[A]

val io = task.toIO                  // Task[A] → IO[A]
io.to[Task]: Task[Int]              // IO[A] → Task[A]

// Future[A] → Task[A] → IO[A] → Task[A] → Future[A]
Task.fromFuture(Future(42)).toIO.to[Task].runToFuture: Future[Int]
```

Horizontal scaling is bidirectional if types are isomorphic, and unidirectional otherwise.

## Vertical Scaling

⼐ With ability to abstract over types, the next step is to abstract over effects.

***Algebra* is to effect what effect is to type.**

One way to abstract over effects is the *tagless final* encoding.

```scala
import java.util.UUID

final case class Snapshot(tenantId: UUID, id: UUID)

trait SnapshotRepo[F[_]] {
  def current(tenantId: UUID): F[Option[Snapshot]]
}
```

Another way is using *Free monads*. {{< marginnote >}}https://typelevel.org/cats/datatypes/freemonad.html{{< /marginnote >}}

```scala
import cats.~>
import cats.free.Free

sealed trait ObjectStoreAlgebra[A]
final case class Put(path: String, bytes: Array[Byte]) extends ObjectStoreAlgebra[Unit]

object ObjectStore {
  type ObjectStore[A] = Free[ObjectStoreAlgebra, A]

  def put(path: String, bytes: Array[Byte]): ObjectStore[Unit] =
    Free.liftF[ObjectStoreAlgebra, Unit](Put(path, bytes))
}

def compiler[F[_]]: ObjectStoreAlgebra ~> F = ???
```

⇄ Algebras can be scaled horizontally by replacing one approach with another.

***Extensible effects* are to algebras what algebras are to effects.** {{< marginnote >}}https://atnos-org.github.io/eff{{< /marginnote >}}

```scala
import cats.Monad
import cats.data.Writer
import org.atnos.eff._
import org.atnos.eff.all._

sealed trait Event
final case class ObjectStored(path: String) extends Event

trait PayloadAlgebra[F[_]] {
  type _effect[R] = F |= R                    // Expect to deal with F effect
  type _writer[R] = Writer[String, ?] |= R    // And the Cats' Writer for logging

  def process[R: _option: _effect: _writer](  // Expect to see some Options
    tenantId:     UUID,
    payload:      Array[Byte],
    snapshotRepo: SnapshotRepo[F],            // Pass tagless final interpreter
    compiler:     ObjectStoreAlgebra ~> F     // Pass free monad compiler
  )(implicit M: Monad[F]): Eff[R, Event] = for {
    _ ← tell("Starting processing")                                 // Cats data type
    maybeSnapshot ← Eff.send(snapshotRepo.current(tenantId))        // Tagless final algebra effect
    snapshot ← fromOption(maybeSnapshot)                            // Built-in Option type
    path = s"${tenantId}/${snapshot.id}"                            // A value
    _ ← Eff.send(ObjectStore.put(path, payload).foldMap(compiler))  // Free monad effect
  } yield ObjectStored(path)
}
```

⇅ Vertical scaling implies both scaling up and scaling down.

☰ Different effects, along with algebra interpreters, are computed in one place. {{< marginnote >}}https://typelevel.org/cats/datatypes/writer.html{{< /marginnote >}}

↓ The API is scaled down being encoded as tagless final `trait PayloadAlgebra[F[_]]`.

↓ Further downscaling is achieved by running the computation.

```scala
type Stack = Fx.fx3[Option, Task, Writer[String, ?]]
val algebra = new PayloadAlgebra[Task] {}

val computation: Task[Option[Event]] = algebra
  .process[Stack](tenantId, payload, snapshotRepo, compiler)
  .runOption
  .runWriterUnsafe[String](println)
  .runAsync
```

⧇ Using the abstract interface, the computation results in nested effects.

⟷ The effect can be scaled horizontally as at the beginning.

{{< mermaid >}}
flowchart BT
  subgraph Effects
  direction LR
  option["Option"]
  cats["Cats Writer"]
  future["Future"]
  task["Monix Task"]
  option -.- cats -.- future <-..-> task
  end
  
  subgraph Algebras
  direction LR
  tf[Tagless final]
  free[Free monad]
  tf <-..-> free
  end
  
  subgraph Stack[Effect Stack]
	eff[Extensible effects]
  end
  
  Effects <-..-> Algebras <-..-> Stack
  Effects <-..-> Stack
{{< /mermaid >}}

↻
