---
title: "Ruby Symbol#to_proc is a lambadass"
description: "Inconsistencies in the Symbol#to_proc implementation across Ruby interpreters"
date: 2013-08-12T23:53:41+03:00
draft: false
---

It all started with the addition of the `to_proc` method to the `Symbol` class, which is admittedly easy to use and looks great. Instead of writing
```ruby
irb > [Object, Kernel, Class].map {|cls| cls.name }
=> ["Object", "Kernel", "Class"]
```

it can be written as
```ruby
irb > [Object, Kernel, Class].map(&:name)
=> ["Object", "Kernel", "Class"]
```

And what it does is simply call the `to_proc` method on the symbol `:name` (which returns a `Proc`), and then convert the proc to a block with the `&` operator (because `map` takes a block, not a proc).

A naive implementation of the `Symbol#to_proc` could look like this:
```ruby
class Symbol
  def to_proc
    Proc.new {|obj, *args| obj.send(self, *args) }
  end
end
```
After fixing cases with arrays and possibly monkey-patched `send` method, you might end up with something like this:
``` ruby
class Symbol
  def to_proc
    Proc.new {|*args| args.shift.__send__(self, *args) }
  end
end
```
It's a well-understood concept with many descriptions available on the Internet. To understand why the `Symbol#to_proc` is a *lambadass* (and what it means), let's move on to the different kinds of Ruby Procs.

## Different kinds of Ruby Procs
As you may know, there are two kinds of Ruby Procs---procs and lambdas. They differ not only in how they check function arity, but also in their appearance when run in irb:
```ruby
irb > Proc.new {}
=> #<Proc:0x007f944a090c50@(irb):1>

irb > lambda {}
=> #<Proc:0x007f944a08ac38@(irb):2 (lambda)>
```

You can see that the `(lambda)` suffix is displayed for lambdas only, and both of them have something like context `(irb):2`. However, it turns out that there is a third kind of procs, which I call the lambadass. But first, let's talk about the lambda scope or context.

## Scope
Procs and lambdas (both instances of the `Proc` class) are closures, just like blocks. The only important thing for us to note is that **they are evaluated in the scope where they are defined or created**.
It means that any block (or proc or lambda) includes a set of bindings (local variables, instance variables, etc.) captured at the moment when it was defined. Simple example demonstrating this in action:
```ruby
irb > x = 1

irb > def z(x)
irb >   lambda { x }
irb > end

irb > lambda { x }.call
=> 1

irb > z(2).call
=> 2
```
As any method definition is a new scope gate, the only known binding with the name `x` inside the method `z` is the method parameter. To visually grab the context of the defined lambda, consider `{}` as the constructor (rather than the `lambda` keyword).

It also means that the lambda defined inside the method body knows nothing about any bindings defined outside of the method scope:
```ruby
irb > z = 1

irb > def x
irb >   lambda { z }
irb > end
```

```
irb > x.call
NameError: undefined local variable or method `z' for main:Object
```
Even though the lambda was called from the top-level scope, it **was defined** inside the method where a binding with the name `z` did not exist. Once the scope or context is captured, it remains the same inside the block no matter where it's called from.

## Lambadass

{{< blockquote cite="http://donderom.com/posts/ruby-symbol-to-proc-is-a-lambadass" footer="Roman Parykin, ‘Ruby Symbol#to_proc is a lambadass’" >}}
Lambadass is a proc or a lambda which looks similar to a standard proc or lambda but behaves differently.
{{< /blockquote >}}

So let's get back to the `Symbol#to_proc`. Usually, it is used instead of a block in methods such as `map`.
Since the `to_proc` method returns a proc, how can we use it standalone like any other proc? Let's do just that:
```ruby
irb > lambda &:name
=> #<Proc:0x007fcfa305dca8>
irb > (lambda &:name).call Class
=> "Class"
```
And lo and behold, it works just as expected! So it looks like it should be identical to
```ruby
irb > lambda {|x| x.name }
=> #<Proc:0x007fcfa30465a8@(irb):8 (lambda)>
irb > lambda {|x| x.name }.call Class
=> "Class"
```
Cool!

But wait a second... Why does the returned `Proc` object look different?  `#<Proc:0x007fcfa305dca8>` instead of  `#<Proc:0x007f944a090c50@(irb):1>`? It's still a `Proc` and a callable object, but it's missing something. Looking at the object representation, I would say it's missing the context. How can we verify this?

### Binding
The bindings captured from the scope where a block is defined are all stored in the `Binding` object. We can get it by calling the `Proc#binding` method:
```ruby
irb > lambda {}.binding
=> #<Binding:0x007fcfa30363b0>
```
One thing we can do with the `Binding` object is to evaluate any binding captured by a block:
```ruby
irb > x = 1
irb > eval('x', lambda {}.binding)
=> 1
```
or
```ruby
irb > x = 1
irb > lambda {}.binding.eval 'x'
=> 1
```
It will raise an exception if a binding with such a name is not defined, but every block (or proc) has an associated binding object.
```ruby
irb > lambda {}.binding
=> #<Binding:0x007fcfa40ab808>
irb > lambda {}.binding.eval 'y'
```
```
NameError: undefined local variable or method `y' for main:Object
```

### Meet the Lambadass
Now let's try to get the binding object of a proc created using the `Symbol#to_proc`:
```ruby
irb > (lambda &:name).binding
```
```
ArgumentError: Can't create Binding from C level Proc
```
Well, there's something wrong with it. It turns out that the `Symbol#to_proc` method is implemented in C in MRI (Matz's Ruby Interpreter which is written in C). So it doesn't make sense to get the context of C level `Proc` object (would be nice though).

What about other interpreters?

### Rubinius

```ruby
rubinius > x = 1
rubinius > lambda {}.binding.eval 'x'
=> 1
rubinius > (lambda &:name).binding.eval 'x'
NameError: undefined local variable or method `x' on name:Symbol.
```
We've got an exception again. This time it says that a binding with the name `x` is not defined. As Rubinius (at least the `Symbol#to_proc`) is written in Ruby itself, let's examine its implementation:
``` ruby symbol19.rb https://github.com/rubinius/rubinius/blob/master/kernel/common/symbol19.rb
def to_proc
  sym = self
  Proc.new do |*args, &b|
    raise ArgumentError, "no receiver given" if args.empty?
    args.shift.__send__(sym, *args, &b)
  end
end
```
The implementation bears a striking resemblance to our initial definition. So what's the problem then? Let's take a look at the error message again:
```
NameError: undefined local variable or method `x' on name:Symbol.
```
Of course there is no variable `x` on the `:name` symbol! The key to understanding it is that the `lambda {}` is defined just here, where the `{}` are, but the `lambda &:name` is defined inside the `Symbol` class in the `to_proc` method, which knows nothing about any bindings defined outside. As for a callable object, its behavior is correct, but its scope is completely different.
To make sense of it, let's take a look at the ```Binding``` object:
``` ruby
rubinius > lambda {}.binding
=> #<Binding:0x179c @variables=#<Rubinius::VariableScope:0x17a0 module=Object method=#<Rubinius::CompiledCode irb_binding file=(irb)>> @compiled_code=#<Rubinius::CompiledCode __block__ file=(irb)> @proc_environment=#<Rubinius::BlockEnvironment:0x17a4 scope=#<Rubinius::VariableScope:0x17a0 module=Object method=#<Rubinius::CompiledCode irb_binding file=(irb)>> top_scope=#<Rubinius::VariableScope:0x1484 module=Object method=#<Rubinius::CompiledCode irb_binding file=.../rubinius/lib/19/irb/workspace.rb>> module=Object compiled_code=#<Rubinius::CompiledCode __block__ file=(irb)> constant_scope=#<Rubinius::ConstantScope:0x17a8 parent=nil module=Object>> @constant_scope=#<Rubinius::ConstantScope:0x17a8 parent=nil module=Object> @self=main>
rubinius > (lambda &:name).binding
=> #<Binding:0x17e0 @variables=#<Rubinius::VariableScope:0x17e4 module=Symbol method=#<Rubinius::CompiledCode to_proc file=kernel/common/symbol19.rb>> @compiled_code=#<Rubinius::CompiledCode to_proc file=kernel/common/symbol19.rb> @proc_environment=#<Rubinius::BlockEnvironment:0x17e8 scope=#<Rubinius::VariableScope:0x17e4 module=Symbol method=#<Rubinius::CompiledCode to_proc file=kernel/common/symbol19.rb>> top_scope=#<Rubinius::VariableScope:0x17e4 module=Symbol method=#<Rubinius::CompiledCode to_proc file=kernel/common/symbol19.rb>> module=Symbol compiled_code=#<Rubinius::CompiledCode to_proc file=kernel/common/symbol19.rb> constant_scope=#<Rubinius::ConstantScope:0x14cc parent=#<Rubinius::ConstantScope:0x14d0 parent=nil module=Object> module=Symbol>> @constant_scope=#<Rubinius::ConstantScope:0x14cc parent=#<Rubinius::ConstantScope:0x14d0 parent=nil module=Object> module=Symbol> @self=:name>
```
In the first case, the module is `Object` and the compiled code is a block in irb. In the second output, the module is `Symbol` and the compiled code is the `to_proc` method in the `kernel/common/symbol19.rb` file.

Of course, if you wrap the `lambda &:name` in another lambda, the scope of the outer lambda will be the `Object` because it is not defined in the `Symbol` anymore. Anyway, the scope of the inner lambda will remain unchanged:
```ruby
rubinius > (lambda &:name).binding
=> #<Binding:0x17e0 ... module=Symbol ...
rubinius > lambda { lambda &:name }.binding
=> #<Binding:0x1854 ... module=Object ...
rubinius > lambda { lambda &:name }.call.binding
=> #<Binding:0x1898 ... module=Symbol ...
```

### JRuby
```ruby
jruby > x = 1
jruby > lambda {}.binding.eval 'x'
=> 1
jruby > (lambda &:name).binding.eval 'x'
=> 1
```
This is the result that almost everyone I asked would expect. No errors, works identically. But if you recall the previously defined `to_proc` method, how scope is defined in Ruby, and Rubinius implementation, this behavior should be considered wrong, even if it seems to be the only one that works without any big surprises.

## Epilogue
There is a proc that sometimes can be a lambda---the same object with the different syntax and behavior from just a proc, but consistent across interpreters. However, with the current implementation of the `Symbol#to_proc`, we have a third behavior of proc that differs across interpreters. I call it lambadass 🕶.
