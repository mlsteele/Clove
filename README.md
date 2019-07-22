# Pastiche
OpenCL doodling.

See the [showcase](https://github.com/mlsteele/pastiche/tree/master/showcase) for some pretty pictures.

One kernel is a copy of [fejesjoco's cool program](https://codegolf.stackexchange.com/questions/22144/images-with-all-colors). I've been working at making a really fast implementation of that. Here's a previous cpu-only [version](https://github.com/mlsteele/joco). This thing can also affect live video!

![swirl](https://github.com/mlsteele/pastiche/blob/master/showcase/512.png)

![conway](https://github.com/mlsteele/pastiche/blob/master/showcase/life.gif)

## Setup

```
$ sudo apt-get install libv4l-dev
# Run in release mode, otherwise it's super slow.
$ cargo run --release
```

## Notes

- [OpenCL channel orders](https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/read_imagef2d.html)
