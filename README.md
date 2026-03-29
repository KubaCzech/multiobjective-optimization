# multiobjective-optimization

To run the container, first build it using:
```docker build -t moo-project .```

Then, run the container using:
```docker run -it -p 8888:8888 -v "$(pwd)":/app moo-project```