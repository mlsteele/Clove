# Pastiche
OpenCL doodling.

See the [showcase](https://github.com/mlsteele/pastiche/tree/master/showcase) for some pretty pictures.

One kernel is a copy of [fejesjoco's cool program](https://codegolf.stackexchange.com/questions/22144/images-with-all-colors). It's been my personal mission to make to make a really fast implementation of that. Here's a previous cpu-only [version](https://github.com/mlsteele/joco).

## Setup

```
$ sudo apt-get install libv4l-dev
# Run in release mode, otherwise it's super slow.
$ cargo run --release
```

## Notes

- [OpenCL channel ordes](https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/read_imagef2d.html)
