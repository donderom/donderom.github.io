---
title: "The arrival of String Interpolation in Scala"
date: 2013-01-06T21:31:10+03:00
draft: false
---

## Scala 2.10 introduces String Interpolation
I always wanted Scala to have something like Ruby string interpolation. It's a relatively small feature, and some may consider it unnecessary syntactic sugar (e.g. Java has no string interpolation at all), but it has always felt just right for Scala. Starting from Scala 2.10, there is a new mechanism for it called String Interpolation (who would have thought!). The documentation can be found [here](http://docs.scala-lang.org/overviews/core/string-interpolation.html) with the corresponding [SIP](http://docs.scala-lang.org/sips/pending/string-interpolation.html). It's a brief overview, so I would recommend reading it through. Although the documentation is clear (there isn't actually much to cover), I would like to highlight a few points anyway.

## String Interpolation is safe
There are three string interpolation methods out of the box: `s`, `f` and `raw`.
Let's take a look at the `s` String Interpolator in action with some variables:
```scala
scala> val x = 1
scala> s"x is $x"
```
```
res1: String = x is 1
```
and with expressions:
```scala
scala> s"expr 1+1 equals to ${1 + 1}"
```
```
res2: String = expr 1+1 equals to 2
```
So it works as expected (at least from Ruby's point of view :)). Let's try to interpolate a variable that doesn't exist:
```scala
scala> s"name doesn't exist $name"
```
```
<console>:8: error: not found: value name
              s"name doesn't exist $name"
                                    ^
```
It just doesn't compile at all! Very nice of the compiler, isn't it? It's not hard to comprehend why it's safe once you have gone through the documentation. If not, we'll address this in the next section.

## String Interpolation is extensible
If you're still wondering what the `s` before the string literal does, the answer is that processed string literal is a code transformation that the compiler converts into a method call `s` on an instance of [StringContext](http://www.scala-lang.org/archives/downloads/distrib/files/nightly/docs/library/index.html#scala.StringContext). In other words, an expression like
```scala
s"x is $x"
```
is rewritten by compiler as
```scala
StringContext("x is ", "").s(x)
```
First of all, it explains why using string interpolation is safe, as demonstrated in the previous section, where passing a nonexistent variable as a parameter for a method call leads to a `not found: value [nonexistent variable here]` error. Secondly, it allows us to define our own string interpolators, potentially reusing the existing ones.

Let's create our own string interpolator, similar to the `s` interpolator, but with debug information added to the resulting string:

```scala
import java.util.Date
import java.text.SimpleDateFormat

object Interpolation {
  val timeFormat = new SimpleDateFormat("HH:mm:ss")
  implicit class LogInterpolator(val sc: StringContext) extends AnyVal {
    def log(args: Any*): String = {
      s"[DEBUG ${timeFormat.format(new Date)}] ${sc.s(args:_*)}"
    }
  }
}

import Interpolation._

val logString = "one plus one is"
log"$logString ${1+1}"
```
```
res8: String = [DEBUG 21:51:50] one plus one is 2
```
The code above uses two new features introduced in Scala 2.10: implicit classes and value classes (by extending `AnyVal`). Since any interpolator is, in fact, a method of the `StringContext` class, we can easily use it in our own code. In the example, we use the `s` function to build the resulting string instead of bothering with its implementation in our new interpolator. So the string interpolation
```scala
log"$logString ${1+1}"
```
will be rewritten by compiler as
```scala
new LogInterpolator(new StringContext("", " ", "")).log(logString, 2)
```
which is a nice combination of the new Scala 2.10 features.

This new technique is useful for writing more readable, safe code, and allows for the extension and combination of existing functionality. The only limitation is that it doesn't work within pattern matching statements, but it is expected to be implemented in the Scala 2.11 release. I would call it String Interpolation with Batteries Included :)
