---
title: "Stretchy architecture"
description: "Stretchy architecture in systems modeling"
date: 2023-07-23T16:33:52+03:00
draft: false
---

{{< newthought >}}Stretchy architecture{{< /newthought >}} is an architectural approach where each dimension is chosen in a way that makes it easier to move in one direction on the decision spectrum than the other. An important condition is to move back to the starting point if possible.

**Asynchronous vs synchronous communication**. Usually presented as a trade-off, in the stretchy architecture style, moving from async to sync is fine and, in practice, it's much easier then the other way round. Starting with async communication by default allows switching to sync and back anytime if needed. Switching from sync to async is not only more difficult but might also be impossible, which locks (*see?*) you into the sync style.

**Eventual vs strong consistency**. Stick to eventual consistency (or any other weaker consistency model), and any part can be made atomic without changing much of the rest. Suggesting to replace the existing strong consistency with weaker models might be dangerous for the messenger. This friction is what makes some people argue about which style is better. Neither is; it's just easier to start with a weaker model and move towards stronger ones if new requirements dictate that.

**Choreography vs orchestrated coordination**. Besides the tight coupling introduced by the mediator, orchestration implementations often lead to synchronous communication, where components must wait for instructions from the orchestrator. On the other hand, a fully choreographed system offers the flexibility to add an orchestrator to facilitate coordination among components if the need arises.

**Immutability vs mutability**. Given a piece of immutable code, it's fine to introduce mutability in certain parts for performance reasons, as long as it remains isolated from the rest of the codebase. It's a perfect illustration of "stretchiness", where we begin with immutable design, stretch it a bit to benefit from lower-level optimizations, and then return back to immutability.

**Stateless vs stateful protocol**. The recurring pattern of increased friction of movement in one direction more than the other is a defining characteristic of the stretchy architecture. Can be confirmed by anyone who has tried to transition from a stateful protocol to a stateless one.

**Schema vs schemaless modeling**. Having a schema for data modeling condones hacks like treating JSON objects as text or bytes. When everything is text, where do you go from there?

**Static vs dynamic typing**. Similar to data modeling but applied to code, statically typed languages provide more guarantees out of the box, and most of them are capable of loosening data representation. One of the useful patterns is to parse data as soon as possible (for example, when reading a file) and work with typed data afterwards. With a map of `any` values by `any` keys, the code is destined to become increasingly problematic.

Thinking in terms of the stretchy architecture results in responsive, highly-available, scalable, fault-tolerant, decoupled, and maintainable systems - essentially for free.
