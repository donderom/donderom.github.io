---
title: "Recipe to slow down your Clojure DSL"
date: 2012-10-12T23:21:11+03:00
draft: false
---

The recipe is intended to be used for making your elegant (in/ex)ternal DSL code much slower than it could be otherwise. It is said that using suggested ingredients correctly can make your code up to 10 times slower without sacrificing elegance and readability. (All the examples are based on a hypothetical implementation of the Porter (also called [Porter 1](http://tartarus.org/~martin/PorterStemmer/)) stemming algorithm).

**Ingredients:** Clojure, understanding of partial function application, and what's under the hood of Clojure data types.

## Step I: Parse strings manually when regular expressions can do the trick
Just try to figure out whether 'Y' letter is a consonant or not.
Don't use regexp. Ever. Did you crave for slower code? Now you have it.
So think twice before using regular expressions with JVM-based languages because it can make your code much faster.

## Step II: Use partial functions everywhere to beautify your DSL
For example, in step two we could use the following rule definitions:
```clojure
{:condition (m > 1) :suffix1 "logi" :suffix2 "log"}
{:condition (m > 1) :suffix1 "ational" :suffix2 "ate"}
;; and so on
```
where `m` is a function that takes two arguments---a function and a number---and returns the partial function which is waiting for the last argument (a given word in our case) and returns the measure of the word. Something like this:
```clojure
(defn m
  [f x]
    (partial #(%1 (measure %3) %2) f x))
```
where `measure` takes the word and produces its measure.
In spite of Clojure's prefix notation, we can easily create readable infix expressions in our DSL. Looks nice, doesn't it? Of course it does! And what's more important for our recipe is that it evaluates slower than it could! Needless to say, we could avoid it by doing it like this:
```clojure
(let [m-cond (m > 1)]
  (apply to-given-word
    {:condition m-cond :suffix1 "logi" :suffix2 "log"}
    {:condition m-cond :suffix1 "logi" :suffix2 "log"}
    ;; and so on ))
```
It's always faster but not as "DSLish" as the initial example.

## Step III: Don't use partial functions with map, filter, etc.
What? "To use or not to use [partial function application]" one might be wondering. As always, the best way to illustrate this is with an example. Imagine that we have some data structure, let's say a vector of letters:
```clojure
(def letters [\a \b \c \d \e])
```
So, if we want to check whether there are any letters in another vector that equal the last letter of the existing vector, we can use the built-in function `some`, like this:
```clojure
(some #(= (peek letters) %) [\f \g \h \i \j])
```
In our case, it is just a great solution because it is slow :).
As you might guess, one of the approaches to make it faster (up to 3 times) is to use partial function instead of our anonymous function:
```clojure
(some (partial = (peek letters)) [\f \g \h \i \j])
```
(And yes, `peek` is faster than `last` for vectors.)

## Step IV: Make use of hashes as often as possible
Here is the example of rule definitions from above:
```clojure
{:condition (m > 1) :suffix1 "logi" :suffix2 "log"}
{:condition (m > 1) :suffix1 "ational" :suffix2 "ate"}
;; and so on
```
You already know what to do (or not) with these `(m > 1)` expressions. I'm sure you know better how to make a DSL slower and the recipe tastier. But it's not what makes the code much much slower. That's our last ingredient---curly braces! Not as additional characters or tokens in our expression, but as a data structure behind it. Good Old Java Hash Map. As using curly braces in Clojure is much more pleasant than creating these cumbersome definitions in Java (`Map<Symbol, Object> rule = new HashMap<Symbol, Object>();` vs `{}`), it creates the illusion of some kind of magic under the hood. The trick is that it's still a Java Hash Map and creating a bunch of them in our DSL is what makes it much much slower than it could be. So the rule is pretty simple---just create as many hashes in the form of curly braces as possible to kill the performance of any DSL.
