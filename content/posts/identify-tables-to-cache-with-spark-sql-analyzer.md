---
title: "Identify tables to cache with Spark SQL Analyzer"
description: "Using Spark Catalyst optimizer to identify tables for caching"
date: 2018-02-28T19:56:54+03:00
draft: false
---

Consider having dozens of Cassandra tables as data sources/sinks, with Spark SQL transformations in between. Sometimes, loading the whole table or a data subset by multiple keys is inevitable. Every optimization applied to the underlying data source and intermediary dataframes goes a long way. One of such optimizations is caching. As the number of transformations grows, the DAG can become unreadable very quickly. It would be nice to be able to check if there are any candidates for caching without having to dig deep into the code. Let's take a look at some examples.

## Use Cases

Basically, any table that is used more than once can be a candidate for caching. Let's create a test table named `t1` (all the code is tested on Spark 2.3.0):

```scala
spark.sql("select 1").createOrReplaceTempView("t1")
```

Now let's create another table that is based on `t1`:

```scala
spark.sql("select * from t1").createOrReplaceTempView("t2")
```

When a Spark action is applied to `t2` table, `t1` is read only once, so it should be fine. What if we create yet another table that is using `t1`?

```scala
spark.sql("select * from t1").createOrReplaceTempView("t3")
```

`t1` is now used twice, by both `t2` and `t3`, so it could potentially benefit from caching. The pitfall is that to determine this, all of the transformations should be analyzed.

Let's create yet another transformation that does not require caching:

```scala
spark.sql("select 2").createOrReplaceTempView("t4")
spark.sql("select * from t4").createOrReplaceTempView("t5")
```

There are some corner cases that would be nice to avoid. In the following example

```scala
spark.sql("select * from (select * from t5) as t1").createOrReplaceTempView("t6")
```

`t1` is an alias, but there is a table with exactly the same name. Therefore,  it should not be mistaken for the actual table.

If a table is already cached, it should be excluded from consideration:

```scala
spark.sql("select 3").createOrReplaceTempView("t7")
spark.table("t7").cache()
spark.sql("select * from t7").createOrReplaceTempView("t8")
spark.sql("select * from t7").createOrReplaceTempView("t9")
```

## In Action

Following is the code covering the aforementioned use cases:

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.plans.logical.SubqueryAlias

object Analyzer {
  type PlanCache = Map[String, LogicalPlan]
  case class Result(parents: Map[String, List[String]], planCache: PlanCache)
    
  def parents(implicit spark: SparkSession): Map[String, List[String]] = {
    def updateCache(table: String, cache: PlanCache): PlanCache =
      if (cache.contains(table)) cache else cache + (table -> analyzedPlan(table))
      
    def subquery(plan: SubqueryAlias): Boolean =
      plan.child.nodeName != "SubqueryAlias" &&
      spark.catalog.tableExists(plan.alias) &&
      !spark.catalog.isCached(plan.alias)

    @annotation.tailrec
    def parentAliases(
      nodes: Seq[LogicalPlan],
      names: List[String],
      planCache: PlanCache
    ): (List[String], PlanCache) = nodes match {
      case (plan @ SubqueryAlias(alias, child)) +: plans if subquery(plan) =>
        val newCache = updateCache(alias, planCache)
        if (plan.sameResult(newCache(alias))) {
          parentAliases(plans, alias :: names, newCache)
        } else {
          parentAliases(plans ++ subplans(plan), names, planCache)
        }
      case plan +: plans =>
        parentAliases(plans ++ subplans(plan), names, planCache)
      case _ => (names, planCache)
    }

    val tables = spark.catalog.listTables.collect
    tables.foldLeft(Result(Map(), Map())) { case (result, table) =>
      val cache = updateCache(table.name, result.planCache)
      val nodes = cache(table.name).children
      val (aliases, newCache) = parentAliases(nodes, Nil, cache)
      val parents = aliases.foldLeft(result.parents) { case (parents, alias) =>
        val children = table.name :: parents.getOrElse(alias, Nil)
        parents + (alias -> children)
      }
      Result(parents, newCache)
    }.parents
  }
  
  def analyzedPlan(table: String)(implicit spark: SparkSession): LogicalPlan =
    spark.table(table).queryExecution.analyzed
    
  def subplans(plan: LogicalPlan): Seq[LogicalPlan] =
    plan.children ++ plan.subqueries

  def showReused(implicit spark: SparkSession): Unit =
    parents.foreach { case (parent, children) =>
      if (children.size > 1)
        println(s"""$parent is used by ${children.mkString(", ")}""")
    }
}
```

To ensure efficiency, the code has been optimized to process every registered table only once. It is worth noting that most of the _cache_ references in the code refer to the internal cache rather than the Spark cache. The idea is this:

* The analyzed logical plan generated by Spark SQL Analyzer is acquired for every table `spark.table(table).queryExecution.analyzed`.
* The child nodes of a query plan are processed recursively to search for `SubqueryAlias` nodes.
* When a `SubqueryAlias` node is found, checks are performed to verify that the table exists, is not cached, is not an alias, etc.
* The goal is to collect all the tables that are used by any other table in order to identify common parent tables.

If we run the code

```
scala> Analyzer.showReused(spark)
t1 is used by t3, t2
```

we can see that:

- `t5` is not a valid candidate and is therefore missing from the results, as expected;
- `t6`, which uses an alias, is also not included in the results;
- `t7` is already cached, so there is no need to consider it for caching.

Thus, only table `t1` is correctly identified as being used by both `t2` and `t3`, making it a suitable candidate for caching.

While Catalyst is primarily used for logical optimizations, its initial analysis phase can also be useful in cases like this, particularly when migrating large volumes of SQL business logic to Spark SQL. In practice, it proved helpful in identifying a few tables that did, in fact, need to be cached.
