# Generator

## How to use

Go to project main directory and run `generator.py` with parameters:
- `num_of_shards`, which defines number of vectors that will be generated - it's identical to number of shards located in total on cloud.
- `num_of_samples`, which defines length of each vectors (they will be all same size).
- `period`, which defines the period of time related to one element in load vector.
- `mean`, which defines mean value; parameter of random normal distribution of vector values (we generate them until all of them are positive values, so be careful if you try to test impossible case)
- `std`, which defines standard deviation; parameter of random normal distribution of vector values


In project main folder run following command:

```
python3.6 generator/generator.py 10 10 2.5 4.0 1.0
```

### Example of expected values

In order to know how generated files should look like, go into `./generator` and find `requests.csv` and `vectors.csv`.

