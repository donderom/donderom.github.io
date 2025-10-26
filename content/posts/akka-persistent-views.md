---
title: "Akka persistent views"
date: 2015-04-10T22:09:55+03:00
draft: false
---

In this post, I'd like to talk about certain aspects of Akka persistence, which is still experimental at the time of writing. The official documentation does a good job of explaining the concept behind Akka persistence and how to make an actor persistent. However, I noticed one thing that can be confusing to many people when they start dealing with akka-persistence module---Persistent Views. Before we discuss it, let's start with the pattern that will help to understand persistent views better.

## CQRS (Command Query Responsibility Segregation)
The idea of the CQRS pattern is to use separate models for reading and updating data, rather than the more classical way of performing CRUD (create, read, update, delete) operations at the same time and place. In terms of CRUD, create, update, and delete are treated as **C**ommands, while read is considered to be a **Q**uery. One of the architectural patterns that CRQS fits with is Event Sourcing (you can read about how Akka persistence implements this idea in the official documentation). Before we delve into persistent views, let's first take a quick look at non-persistent actors.

## CQRS with Non Persistent Actors
Let's imagine that there is an actor that keeps transactions in its internal state. We can use the same actor to store new transactions and read them. One of the issues with such a design is the need for having completely different representations for display. For instance, we may want to display all invalid transactions and the current balance for a particular account on the administration panel. If the actor is in the middle of updating its state, it cannot be queried, which makes the system less responsive. To separate commands and queries, we will most likely end up with more than one actor, each with its own state reflecting changes from the main actor. This setup will work pretty well until the main actor, with all the transactions, goes down. Once this happens, the state is lost forever. The other actors can still function but the data will stay out of sync.

## Persistent Views and CQRS
To make it clear, Persistent View represents the Query part of CQRS. Persistent actors are responsible for managing state (persisting events, deleting messages, restoring from the journal, saving the snapshots, etc.). Persistent views poll messages directly from the journal of a persistent actor, without being coupled to the actor itself. All they have to know is the identifier of the corresponding persistent actor. If that persistent actor crashes, it does not affect persistent views as they can still serve data from the journal. Once the persistent actor is recovered, persistent views will eventually become consistent with it again. Moreover, a persistent view can utilize other data sources to optimize data presentation. Basically, it comes down to this: use persistent actors to handle commands and persistent views to handle queries.

## Persistent Views in Action
The working demo project can be found [here](https://github.com/donderom/monadzoo-persistent-view). The source code consists of three files: one for the persistent actor and the other two for the persistent views. The `TransactionActor` is a persistent actor responsible for persisting transactions. All the machinery in the source code is primarily focused on maintaining and recovering state. The `InvalidTransactionsView` is a persistent view that shows all invalid transactions. You can see how it's decoupled from the `TransactionActor`. Another persistent view is the `BalanceView`. Instead of keeping a list of transactions, it keeps a map where it stores the current amount per account. In a traditional approach, both persistent views would be methods of the same object. With separate views, even if the persistent actor and one of the views die, the remaining view will still be able to process queries.

I think Persistent Views play an important role when being used with event sourcing. They help to follow the CQRS pattern more easily than using regular actors would. I would even argue that they are just as important as persistent actors. The official documentation would be more helpful if it provided more information about them.
I hope that the analogy with CQRS can help develop a better understanding of their role.
