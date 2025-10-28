---
title: "Functional builder pattern"
description: "Functional composition with the Builder Pattern in Scala"
date: 2017-03-11T15:54:04+02:00
draft: false
---

The builder pattern is used for constructing objects as a series of steps.

Let's use a database query builder as an example.

```scala
import scala.util.{Failure, Success, Try}

case class Query(
  table: String,
  keyspace: Option[String] = None,
  select: List[String] = Nil,
  where: List[String] = Nil,
  limit: Option[Int] = None
) {
  def build: Try[String] =
    for {
      table <- renderTable
      select <- renderSelect
      where <- renderWhere
      limit <- renderLimit
    } yield s"select $select from $table $where $limit".trim

  def keyspace(name: String): Query =
    this.copy(keyspace = Option(name).filterNot(_.isEmpty))

  def select(col: String): Query =
    this.copy(select = col :: select)

  def where(clause: String): Query =
    this.copy(where = clause :: where)

  def limit(l: Int): Query =
    this.copy(limit = Option(l))

  private def renderTable: Try[String] =
    if (table.isEmpty) Failure(error("Table name cannot be empty"))
    else Success(keyspace.filterNot(_.isEmpty).fold(table)(k => s"$k.$table"))

  private def renderSelect: Try[String] =
    Success(if (select.isEmpty) "*" else select.mkString(", "))

  private def renderWhere: Try[String] =
    Success(if (where.isEmpty) "" else s"""where ${where.mkString(" and ")}""")

  private def renderLimit: Try[String] =
    limit.fold(Try("")) { l =>
      if (l < 1) Failure(error("Limit must be strictly positive"))
      else Success(s"limit $l")
    }

  private def error(msg: String): IllegalArgumentException =
    new IllegalArgumentException(msg)
}
```

With some basic validation, there are strings everywhere for the sake of simplicity, as it's not about type-safety now. The query can be constructed as follows:

```scala
val query = Query(table = "test")
  .select("col")
  .where("col2 = 1")
  .limit(1)
  .keyspace("stg")
  .build

scala> scala.util.Try[String] = Success(select col from stg.test where col2 = 1 limit 1)
```

## Functional approach

{{< newthought >}}The gist{{< /newthought >}} of the functional approach is making each transformation step autonomous by disentangling it from the builder. In other words, each step becomes a first-class citizen function.

```scala
object Query {
  def keyspace(name: String): Query => Query = _.keyspace(name)
  def select(col: String):    Query => Query = _.select(col)
  def where(clause: String):  Query => Query = _.where(clause)
  def limit(value: Int):      Query => Query = _.limit(value)
}
```

Each step is an explicit `Query => Query` function that delegates the construction logic to the builder. The benefits of such wrapping include:

* The original builder can be used without any changes.
* The construction can be composed in multiple steps (e.g. query optimization based on client request).
* Conditional steps become possible (try to add the `where` step conditionally in the example above).
* Parts of the construction can be done elsewhere and passed for composition.
* New steps can be created without the original builder being aware of them.

The initial example can be written like this:

```scala
import Query._

val baseQuery = Query(table = "test")

val steps = select("col") andThen where("col2 = 1") andThen limit(1) andThen keyspace("stg")
steps(baseQuery).build

scala> scala.util.Try[String] = Success(select col from stg.test where col2 = 1 limit 1)
```

One of the potential drawbacks is chain ergonomics with all these `andThen`s. It can be improved by adding a shorter alias to the `Function1` type:

```scala
implicit class FunctionAlias[A, B](f: Function1[A, B]) {
  def ~[C](g: B => C): A => C = f andThen g
}

val steps = select("col") ~ where("col2 = 1") ~ limit(1) ~ keyspace("stg")
steps(baseQuery).build

scala> scala.util.Try[String] = Success(select col from stg.test where col2 = 1 limit 1)
```

Another example of function composition is creating a step from a list of columns:

```scala
val identity = (q: Query) => q
val cols = List("col1", "col2", "col3").foldLeft(identity)(_ ~ select(_))
val steps = keyspace("namespace") ~ cols
steps(baseQuery).build

scala> scala.util.Try[String] = Success(select col3, col2, col1 from namespace.test)
```

## Arrows

If you already use [Cats](https://typelevel.org/cats) or need additional functionality, the same can be accomplished with the [Arrow](https://typelevel.org/cats/typeclasses/arrow.html) type class.

```scala
import cats.implicits._

val identity = cats.arrow.Arrow[Function1].id[Query]
val cols = List("col1", "col2", "col3").foldLeft(identity)(_ >>> select(_))
val steps = keyspace("namespace") >>> limit(1) >>> select("col") >>> cols
steps(baseQuery).build

scala> scala.util.Try[String] = Success(select col3, col2, col1, col from namespace.test  limit 1)
```

Note that `>>>` is identical to ours `~` as it's just an alias for `andThen`.

The whole construction can be built from any collection(s) of steps:

```scala
List(
  keyspace("namespace"),
  limit(1),
  select("col"),
  cols
).foldLeft(identity)(_ >>> _)(baseQuery).build
```

The unobtrusive functional layer on top of any builder is what makes it an elegant approach in my opinion.
