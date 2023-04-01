---
title: "Scala Function object to the rescue"
#date: 2015-04-01T20:43:55+03:00
draft: false
---

The `Function` object has been there since Scala version 1.0. It provides a bunch of utility methods for dealing with higher-order functions. Despite its simplicity and usefulness, I have found that not only do many novice developers not use them, but they are also unaware of their existence. In this post, I would like to bring to your attention some of the functions I find helpful in certain scenarios.

### `chain`

To get started, let's introduce two extremely simple functions from `Int` to `Int`. One of them, let's call it `inc`, will increase a number by one, while the other one, `double`, with multiply a number by two.{{< marginnote >}}We could use a shorter definition form such as `val inc: Int => Int = _ + 1` for both functions.{{< /marginnote >}}
```scala
scala> val inc: Int => Int = x => x + 1
scala> val double: Int => Int = x => x * 2
```
So what do we do when we have a sequence of such functions that we want to combine? One of the most popular options in my practice is the following:
``` scala
scala> List(inc, double, inc) reduce (_ andThen _)
```
This piece of code takes a sequence of functions and combines them using the `andThen` method starting with the first one in the list and returns the resulting function from `Int` to `Int`. It is by no means a bad code. Let's see how we can simplify it by using the `chain` function from the `Function` object:
``` scala
scala> Function.chain(List(inc, double, inc))
```
It does exactly the same as the previous snippet but it might be more intuitive. If we first import all the functions from the `Function` object, it will be dead simple:
``` scala
scala> import Function._
scala> chain(List(inc, double, inc))
```
If you want to use the `compose` function instead of the `andThen`, you can still achieve the same with the `chain` by reversing the sequence first:
``` scala
scala> chain(List(inc, double, double).reverse)
```

### `tupled`

Imagine there is a tuple of two elements, id and name, representing a user wrapped into `Option`:
``` scala
scala> val user: Option[(Int, String)] = Some(1, "Bob")
```
And a function, let's call it `auth`, which accepts id and name as **two separate arguments**:
``` scala
scala> val auth: (Int, String) => Boolean = (id, name) => id == 1 && name == "Bob"
```
We can't just map the `auth` function over the `user` because the former expects two arguments while our user is a tuple with two elements. One option would be to extract the id and name and pass them as separate arguments to the `auth` function but it requires some boilerplate to write. That's when we can use the `tupled` function from the `Function` object. It takes a function with, let's say, two arguments, and converts it into a function that takes a tuple of two elements with the same types as the original arguments. That's exactly what we need to map the `auth` function over the user:
``` scala
scala> user map auth.tupled
res4: Option[Boolean] = Some(true)
```
There are `tupled` functions defined for functions with arity from two to five inclusive. I find it convenient as I don't use tuples of more than two or three elements that often.

Technically, in the previous example, the `tupled` was called on the function itself (as it's defined both in the `Function` object and the `Function*` traits). Another "trick" that can be useful is mapping over a hash map in what many people would say a natural way. (It also demonstrates the usage of the `tupled` function from the `Function` object and not defined in the `Function*` trait.) In Scala, you can't write code like this:
``` scala
scala> val m = Map(1 -> "first", 2 -> "second")
scala> m map { (k, v) => s"$k:$v" }
```
```
<console>:12: error: missing parameter type
Note: The expected type requires a one-argument function accepting a 2-Tuple.
      Consider a pattern matching anonymous function, `{ case (k, v) =>  ... }`
              m map { (k, v) => s"$k:$v" }
                       ^
<console>:12: error: missing parameter type
              m map { (k, v) => s"$k:$v" }
```
What many developers would do is something like this:
``` scala
scala> m map { case (k, v) => s"$k:$v" }
res5: scala.collection.immutable.Iterable[String] = List(1:first, 2:second)
```
You can achieve the same using the `tupled` function:
``` scala
scala> m map tupled { (k, v) => s"$k:$v" }
res6: scala.collection.immutable.Iterable[String] = List(1:first, 2:second)
```
Maybe not so useful but it helps to understand its application.

### `unlift`

To illustrate the usage of the `unlift`, let's write a function that takes an `Int` and returns `Some(x)` if `x` is equal to or greater than zero, and `None` otherwise:
``` scala
scala> val f: Int => Option[Int] = Option(_) filter (_ >= 0)
```
The `unlift` function turns an `A => Option[B]` function into a `PartialFunction[A, B]`. It allows us to use our function in any place where a partial function is required. To make it clear, this how we can use our function to filter out negative integers from a list:
``` scala
scala> import Function._
scala> List(-1,0,1) collect unlift(f)
res7: List[Int] = List(0, 1)
```
I don't use this on a daily basis but there are cases like this where it comes very handy.

_Bonus_: there is an opposite function defined in the `PartialFunction` called `lift`. To see how it is related to the `unlift` function, the following equation is always true:
``` scala
scala> f == unlift(f).lift
res8: Boolean = true
```

### `uncurried/untupled`

These two functions are the opposite to the `curried` and the `tupled` functions, respectively. I haven't seen them being used as frequently as the ones described above.

Although there is nothing new, I believe that we often overlook some of the useful API provided by the Scala standard library. I hope that this refresher is on time and can save you a few lines of code every now and then.
