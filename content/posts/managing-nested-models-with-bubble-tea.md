---
title: "Managing nested models with Bubble Tea"
description: "Model Stack architecture to compose nested models with Bubble Tea framework for Go"
date: 2025-07-09T20:00:02+02:00
draft: false
features: ["diag"]
---

{{< epigraph pre="r/golang, Reddit" >}}
I don't get Bubbletea
{{< /epigraph >}}

Judging by other comments and similar posts, this experience is not unique.

In this post, I’d like to share an intuition behind model composition using the [Bubble Tea](https://github.com/charmbracelet/bubbletea) TUI framework for Go, with the main focus on model nesting. While I assume a basic understanding of the Bubble Tea framework, let’s start by going over what a model is.

## The Elm Architecture

The Bubble Tea is based on [The Elm Architecture](https://guide.elm-lang.org/architecture){{< sidenote >}}The acronym TEA, short for The Elm Architecture, is what gives Bubble Tea its name.{{< /sidenote >}}, where the model is the central piece of application state, representing the entire state of the application at any given point in time. The other two components are:

* The `Update` function that modifies the `model` in response to messages.
* The `View` function that uses the `model` to determine what to show to the user.

In Elm, the `model` itself isn’t a formal concept, but a common pattern used to structure different states of the application. In practice, it’s a data structure that holds everything needed to render itself.

## Nested models

And that’s where the official tutorial ends, without even mentioning how to handle nested states, or, in other words, nested models. Let’s take a look at an  example called [Composable Views](https://github.com/charmbracelet/bubbletea/tree/main/examples/composable-views) from Bubble Tea’s [examples](https://github.com/charmbracelet/bubbletea/tree/main/examples/#composable-views) folder. The model, which composes two other models (timer and spinner), looks like this:

```go
type sessionState uint

type mainModel struct {
	state   sessionState
	timer   timer.Model
	spinner spinner.Model
	//...
}
```

The two models, `timer.Model` and `spinner.Model`{{< sidenote >}}All `xyz.Model`s are part of the Charm [bubbles](https://github.com/charmbracelet/bubbles) project.{{< /sidenote >}}, are nested within the `mainModel`. The `sessionState` is used to determine which model is currently active. Both the `Update` and `View` functions then branch their logic based on the current `sessionState`, either routing messages to the appropriate model(s) or rendering the active model, respectively.

{{< mermaid >}}
flowchart TD
    mainModel --> timer.Model
    mainModel --> spinner.Model
{{< /mermaid >}}

As the number of nested models grows, we may want to move some of them into separate models, thereby creating a tree model structure.

```go
type subModel struct {
	time    timer.Model
	spinner spinner.Model
}

type mainModel struct {
	state    sessionState
	list     list.Model
	subModel subModel
}
```

{{< mermaid >}}
flowchart TD
    Main[mainModel] --> list.Model
		Main --> Sub[subModel]
    Sub --> timer.Model
    Sub --> spinner.Model
{{< /mermaid >}}

No matter the size of the model tree, the `mainModel` remains the top-level model, responsible for routing messages to the child models and view rendering with the content of the child’s `View` methods. For example, this is how updating the `timer.Model` looks like:

{{< mermaid >}}
flowchart TD
    Main[mainModel] --> list.Model
    Main -- calls Update of the subModel --> Sub[subModel]
    Sub -- it’s up to subModel to\n update the timer.Model --> timer.Model
    Sub --> spinner.Model
    classDef active fill:#F1FADF
    class Main,Sub,timer.Model active
{{< /mermaid >}}

There is nothing wrong with this approach. In fact, this is the recommended way to handle the nested models, as demonstrated in more detail in the [“Bubble Tea Nested Models”](https://www.youtube.com/watch?v=uJ2egAkSkjg) video. It makes perfect sense if the child models are part of the current view layout{{< sidenote >}}Generalizing the modeling of components (or [bubbles](https://github.com/charmbracelet/bubbles)) with the current Go (1.25) type system is challenging. Although referred to as models, bubbles aren’t true models in the strict sense - they don’t implement the `tea.Model` interface and they provide additional API. One possible workaround would be to make their API event-driven, but this would come at the cost of event loop performance.{{< /sidenote >}}. However, not all application models are like that. Some models represent entirely new screen views and have nothing to do with the main or previous model. Does this change the handling of nested models, as described above? Not at all. The main model is still responsible for tracking the current model and rendering the appropriate view in place of its own.

To recap, the top-down approach is characterized by the following attributes:
* Keeping track of the application state.
* Routing messages to the appropriate models in the `Update` function based on the current state.
* Composing or delegating the view in the `View` function based on the current state.

## Independent nested models

Let’s bring in one more example to the picture, such as the [Credit Card Form](https://github.com/charmbracelet/bubbletea/tree/main/examples/credit-card-form). The main model will consist of `list.Model` and two submodels: `SubModel1` for *Composable Views* and `SubModel2` for the *Credit Card Form*. The main view might show a list of available examples (two, in this case). When the user selects an example, the corresponding view is shown instead of the list. Both `SubModel1` and `SubModel2` are completely independent of each other and of the main model. In this scenario, the `MainModel`’s `Update` and `View` functions should handle at least three states. Even if the submodel’s view is self-contained, the `MainModel`’s view should still be updated.

{{< mermaid >}}
flowchart TD
  Main[MainModel] --> list.Model
  Main --> sub
  Main --> sub2
  subgraph sub[Composable Views]
    direction TB
    Sub[SubModel1] --> timer.Model
    Sub --> spinner.Model
  end
  subgraph sub2[Credit Card Form]
    direction TB
    Sub2[SubModel2] --> Inputs["[]textinput.Model"]
  end
{{< /mermaid >}}

*An independent model is one that contains all the information necessary to render a screen and handle user interactions.* Examples of independent models include the main model and both submodels. In contrast, models like `list.Model` or `textinput.Model` are usually not independent, as they require additional models to manage user interaction. Some models can serve both roles simultaneously, such as both submodels. If a submodel is shown upon item selection, it is considered independent. However, if it’s shown alongside the list, it becomes part of the main model’s view and is therefore not independent.

The steps for adding an independent model are closely tied to the top-down approach:
1. Add a new state to the main model.
2. Extend the main model’s `Update` function to handle the new state and route messages to the new model.
3. Update the main model’s `View` function to render the new model’s view based on the updated state.

Let’s see how to get rid of all these steps with the **Model Stack** architecture.

## Model Stack architecture

The concept behind the Model Stack architecture is to restructure a tree of nested models into a stack of independent ones. In the example above, the model tree can be decomposed into three independent models, stacked on top of one another as needed.
{{< mermaid >}}
flowchart TB
  subgraph sub2[Credit Card Form]
    direction TB
    Sub2[SubModel2] --> Inputs["[]textinput.Model"]
  end
  subgraph sub[Composable Views]
    direction TB
    Sub[SubModel1] --> timer.Model
    Sub --> spinner.Model
  end
  subgraph main[Application Entry]
    direction TB
    Main[MainModel] --> list.Model
  end
{{< /mermaid >}}

Initially, the model stack contains only the entry model. Each model is responsible for managing its own state and knows how to render itself - it behaves just like any other component. How is another model displayed? It’s pushed onto the stack, becoming the current model. The model stack controller handles model initialization, routes messages to the current model, and displays the current model’s view.

{{< mermaid >}}
flowchart LR
  subgraph stack0[Model Stack]
    subgraph main0[Application Entry]
      direction TB
      Main0[MainModel] --> list.Model
    end
    Label1[HEAD] .-> main0
  end
  subgraph stack[Model Stack]
    subgraph sub[Composable Views]
      direction TB
      Sub[SubModel1] --> timer.Model
      Sub --> spinner.Model
    end
    subgraph main[Application Entry]
      direction TB
      Main[MainModel] --> List[list.Model]
    end
    Label2[HEAD] .-> sub
  end
  stack0 -- put model \non the stack --> stack
  classDef active fill:#F1FADF
  classDef lbl fill:#FFFFFF,stroke:#FFFFFF,font-style:italic
  class main0,sub active
  class Label1,Label2 lbl
{{< /mermaid >}}

The model stack controller takes over routing for nested models - replacing the need for coordination required in the top-down approach within the main model’s `Update` and `View` functions - thus eliminating the steps for adding an independent model. The controller plays the role of the main model and intercepts all commands, but it does not need session state. All that’s left for models is to notify the controller.

### Bubblon

Let’s see the benefits of this approach using the [bubblon](https://github.com/donderom/bubblon) library, which implements the Model Stack architecture for the Bubble Tea framework.

**Event-driven**

Since the models are supposed to be independent, they cannot be accessed directly because the stack is encapsulated by the controller. The only way to interact with the controller is by sending commands. This allows any model to communicate with the controller without holding a direct reference to it.

**Minimalistic**

At its core, the Model Stack architecture is about simplification. For example, the `bubblon.Open` command will initialize the provided model, make it the current one, and redirect messages to it. All of this is accomplished in a single line of code{{< sidenote >}}Internally, there is a stack of models, but semantic API - using commands like `Open` and `Close` - makes more sense in the context of TUI apps.{{< /sidenote >}}:

```go
func (m MainModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// ...
	return m, bubblon.Open(SubModel1.New())
}
```

When we’re done with the submodel and want to return to the main model:

```go
func (m SubModel1) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// ...
	return m, bubblon.Close
}
```

Following its event-driven design, `bubblon` will send the `MainModel` the `bubblon.Closed` message, indicating that the submodel has been closed.

The API remains unobtrusive by introducing no new interfaces or concepts. Models are expected to implement the existing `tea.Model` interface, and all interactions are performed using Bubble Tea commands. The controller itself is a `tea.Model` and can be used like any other model.

**Modular**

One advantage of using independent models is *dynamic composition*. Any model can become the current one. Model `A` can open model `B`, which in turn can open model `C` - without model `A` knowing anything about it. Or `C` → `B` → `A`. No state, no cry. While Go prevents circular dependencies at the package level, it’s still possible to pass a constructed instance or a function to initialize a parent model.

**Resource-efficient**

By default, models are released and made available to the garbage collector once closed by the controller, rather than being kept indefinitely in the main model’s internal state. In addition, more efficient commands are available for replacing the top model - or all models - with a new one, without relying on a `Close/Open` sequence: `bubblon.Replace` and `bubblon.ReplaceAll`{{< sidenote >}}The full API reference can be found on [go.dev](https://pkg.go.dev/github.com/donderom/bubblon).{{< /sidenote >}}.

Since the controller is not responsible for model creation, it is not limited to dynamic composition. Otherwise it would be a waste of resources to recreate a model that can be instantiated just once. This is where a *hybrid approach* to model caching comes in. A submodel that is better suited for one-time creation can be embedded for repeated use, with all the other benefits of the Model Stack architecture preserved.

```go
type model struct {
	submodel tea.Model
}

// somewhere
// m := model{&submodel{}}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Maybe reset submodel's state first
	return m, bubblon.Open(m.submodel)
}
```

Without any session state or command coordination, the submodel is cached by the main model and can be reused at any time. The goal of the Model Stack architecture is to enable both dynamic composition and model caching with minimal code changes.

To summarize, there are at least three progressive levels for building Bubble Tea applications using nested models with [bubblon](https://github.com/donderom/bubblon):

* **Level 1**: Use top-down approach to build a model from individual components.
* **Level 2**: Use the Model Stack architecture to compose an application from independent models.
* **Level 3**: Use a hybrid approach with the Model Stack architecture to maximize flexibility and performance.

Check out [bubblon](https://github.com/donderom/bubblon) on GitHub for more practical examples.
