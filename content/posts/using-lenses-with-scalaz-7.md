---
title: "Using Lenses with Scalaz 7"
date: 2012-11-18T18:49:05+03:00
draft: false
---

The release of [Scalaz](https://github.com/scalaz/scalaz) 7 is getting closer and closer every day, and many projects are migrating to new version of this awesome library. This post is meant to help with migration of Scalaz' lenses. Assuming that you are already familiar with lenses, some basic changes and additions will be demonstrated through examples.

For our examples we will use the following simple model:
```scala
case class Address(city: String, zip: Int)
case class User(name: String, address: Address)
```

Reminder: out of the box field access/update boilerplate:
```scala
val user = User("Nad", Address("Sallad", 79071))
val updatedUser = user.copy(name = "Evets")
val updatedAddress = user.copy(address = user.address.copy(city = "Revned"))
```

The `Lens` type signature is different (if not completely rewritten :)) in Scalaz 7.

What the `Lens` is in Scalaz 6:
```scala
case class Lens[A,B](get: A => B, set: (A,B) => A)
```

And Scalaz 7:
```scala
type Lens[A, B] = LensT[Id, A, B]

object Lens extends LensTFunctions with LensTInstances {
  def apply[A, B](r: A => Store[B, A]): Lens[A, B] =
    lens(r)
}

// A The type of the record
// B The type of the field
sealed trait LensT[F[+_], A, B] {
  def run(a: A): F[Store[B, A]]

  def apply(a: A): F[Store[B, A]] =
    run(a)
  ...
}

type Store[A, B] = StoreT[Id, A, B]
// flipped
type |-->[A, B] = Store[B, A]
object Store {
  def apply[A, B](f: A => B, a: A): Store[A, B] = StoreT.store(a)(f)
}
```

The `Lens` constructor has been completely changed in Scalaz 7. Despite this, there is a `Lens.lensu` constructor which takes the same arguments as the old one, but in a flipped order. So let's create lenses for user name, address, and zip code (through address):
```scala
val nameL: Lens[User, String] = Lens.lensu((u, newName) => u.copy(name = newName), _.name)
val addrL: Lens[User, Address] = Lens.lensu((u, newAddr) => u.copy(address = newAddr), _.address)
val zipL: Lens[Address, Int] = Lens.lensu((a, newZip) => a.copy(zip = newZip), _.zip)
```

Now we can read and update user fields with appropriate lenses as before:
```scala
scala> nameL.get(user)  // reading user name (nameL(user) is no longer identical to nameL.get(user))
res0: scalaz.Id.Id[String] = Nad

scala> addrL.set(user, Address("Empty", 0))  // updating user address
res1: scalaz.Id.Id[User] = User(Nad,Address(Empty,0))
```

In contrast to `set`, the parameters of the `mod` function have been flipped:
```scala
scala> zipL.mod((1+), user.address)  // modifying user zip code through address (user.address)
res2: scalaz.Id.Id[Address] = Address(Sallad,79072)
```

There is a useful addition to the `mod` function called `=>=`:
```scala
scala> val partialMod = nameL =>= (_ + "!")
partialMod: User => scalaz.Id.Id[User] = <function1>

scala> partialMod(user)
res3: scalaz.Id.Id[User] = User(Nad!,Address(Sallad,79071))
```

In addition to the `compose` and `andThen` functions, there are aliases `<=<` and `>=>` respectively:
```scala
scala> zipL compose addrL
res4: scalaz.LensT[scalaz.Id.Id,User,Int] = scalaz.LensTFunctions$$anon$5@51557949

scala> zipL <=< addrL // just an alias to the compose function
res5: scalaz.LensT[scalaz.Id.Id,User,Int] = scalaz.LensTFunctions$$anon$5@3f1cf257

// composing two lenses (Lens[User, Address] andThen Lens[Address, Int] = Lens[User, Int])
scala> val zipThroughUserL = addrL andThen zipL
zipThroughUserL: scalaz.LensT[scalaz.Id.Id,User,Int] = scalaz.LensTFunctions$$anon$5@5c921914

// modifying user zip code through user itself with composed lenses
scala> zipThroughUserL.mod((_ - 1), user)
res6: scalaz.Id.Id[User] = User(Nad,Address(Sallad,79070))

scala> (addrL >=> zipL).mod((_ - 1), user) // the same as two previous lines
res7: scalaz.Id.Id[User] = User(Nad,Address(Sallad,79070))
```

Homomorphism of lens categories lets us get a value of type `Option[B]` (where `B` is the type of the field):
```scala
// A homomorphism of lens categories
scala> nameL get user  // result is of type String
res8: scalaz.Id.Id[String] = Nad

scala> ~nameL get user  // but here the result is of type Option[String]!
res9: scalaz.Id.Id[Option[String]] = Some(Nad)
```

Using lens as a State monad (including `map` and `flatMap` as `>-` and `>>-` respectively):
```scala
// Set the portion of the state viewed through the lens and return its new value
scala> val zipState = for {
     |   x <- zipL
     |   _ <- zipL := x + 1
     | } yield x
zipState: scalaz.StateT[scalaz.Id.Id,Address,Int] = scalaz.StateT$$anon$7@346d9067

scala> zipState.run(user.address)
res10: (Address, Int) = (Address(Sallad,79072),79071)

scala> zipState.eval(user.address)  // discard the final state
res11: scalaz.Id.Id[Int] = 79071

// Modify the portion of the state viewed through the lens and return its new value
scala> (nameL %= (_ + "!")) run user
res12: (User, String) = (User(Nad!,Address(Sallad,79071)),Nad!)

// Modify the portion of the state viewed through the lens, but do not return its new value
scala> (nameL %== (_ + "!")) run user
res13: (User, Unit) = (User(Nad!,Address(Sallad,79071)),())

// Map the function over the value under the lens, as a state action
scala> (nameL >- (_.toUpperCase)) run user  // >- is an alias for map
res14: (User, java.lang.String) = (User(Nad,Address(Sallad,79071)),NAD)

// Bind the function over the value under the lens, as a state action
val upNameL: Lens[User, String] = Lens.lensu((u, newName) => u.copy(name = newName.toUpperCase), _.name.toUpperCase) // yet another lens for user name

scala> (nameL >>- (_ => upNameL)) run user  // >>- is an alias for flatMap
res15: (User, String) = (User(Nad,Address(Sallad,79071)),NAD)

// Sequence the monadic action of looking through the lens to occur before the state action
scala> nameL ->>- upNameL run user  // uses flatMap (>>-) for sequencing monadic actions
res16: (User, String) = (User(Nad,Address(Sallad,79071)),NAD)
```

Examining the examples above, it's not hard to notice that there is a bunch of new methods and convenient aliases, especially for using lens as a state monad. Lenses have become even easier to use now, when the underlying type complexity is under the hood.
