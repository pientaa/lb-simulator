# lb-simulator

## Table of contents

- [lb-simulator](#lb-simulator)
  * [Table of contents](#table-of-contents)
  * [Generator](#generator)
      - [Run](#run)
      - [Tasks distribution](#tasks-distribution)
  * [Shard allocator](#shard-allocator)
  * [Simulator](#simulator)
  * [Example of generated data](#example-of-generated-data)
    + [Requests](#requests)
    + [Load vectors](#load-vectors)
    + [Shard allocated](#shard-allocated)

 
## Generator

Generator takes following parameters:
- `num_of_shards`, which defines number of vectors that will be generated - it's identical to number of shards located in total on cloud.
- `num_of_samples`, which defines nomber of periods that will be considered when generating tasks.
- `period`, which defines the amount of time related to one element of load vector.
- `shape`, which defines &alpha; parameter of gamma distribution from which number of tasks per period will be drawn.
- `size`, which defines &beta; parameter of gamma distribution from which number of tasks per period will be drawn.
 
#### Run

Generate requests and load vectors using following function from `generator` module:
```python
def generator(num_of_shards, num_of_samples, new_period, shape, scale)
```

#### Tasks distribution

Tasks are drawn from gamma distribution, but to be precise - as far as `shape` parameter is scalar -  it's Erlang distribution, whose mean and standard deviation can be calulated with following equations:

 &mu; = &alpha; * &beta; </br>
 &sigma; =  &beta; * √&alpha;


## Shard allocator

Shard allocator is able to allocate shards on nodes according to one of three algorithms:
- `random` - evenly distribute shards on nodes with random order
- `sequential` - evenly distribute shards on nodes with sequential order
- `SALP` - Shards Allocation based on Load Prediction

```python
def shard_allocator(shards, nodes, algorithm_name):
```

- `shards` - number of shards
- `nodes` - number of nodes
- `algorithm_name` - one of listed above algorithm `random`/`sequential`/`SALP`


## Simulator


## Example of generated data


```python
generator(5, 10, 5, 2, 2)
```

### Requests

<details>
 <summary>
Example of requests
 </summary>

|  id  |timestamp|shard|load |
|------|---------|-----|-----|
|0     |1.23     |3    |3.266|
|1     |3.27     |4    |3.861|
|2     |3.58     |4    |7.198|
|3     |5.02     |3    |3.022|
|4     |5.305    |4    |1.727|
|5     |5.31     |5    |1.729|
|6     |5.63     |1    |5.434|
|7     |6.155    |1    |7.069|
|8     |6.835    |2    |3.011|
|9     |7.695    |4    |12.173|
|10    |8.525    |1    |4.723|
|11    |11.09    |5    |11.399|
|12    |12.67    |5    |1.523|
|13    |14.085   |1    |1.942|
|14    |17.365   |1    |2.025|
|15    |19.405   |3    |11.537|
|16    |20.25    |3    |1.623|
|17    |24.995   |3    |2.994|
|18    |25.795   |3    |3.882|
|19    |25.97    |3    |1.371|
|20    |29.84    |4    |3.541|
|21    |32.645   |3    |0.743|
|22    |34.245   |1    |1.99 |
|23    |38.515   |1    |5.539|
|24    |38.555   |2    |9.327|
|25    |41.55    |4    |6.34 |
|26    |44.755   |1    |9.249|
|27    |45.22    |4    |3.795|
|28    |45.275   |4    |6.259|
|29    |49.99    |2    |2.705|

</details>

### Load vectors

<details>
 <summary>
Example of load vectors
 </summary>

| [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] | [9] | [10] | [11] |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|0.0  |1.938|1.691|0.61 |0.0  |0.0  |0.151|0.544|0.86|1.0  |1.64 |
|0.0  |0.602|0.0  |0.0  |0.0  |0.0  |0.0  |1.289|0.0 |1.61 |0.539|
|0.653|0.604|0.0  |1.119|1.326|1.648|2.033|0.0  |0.0 |0.0  |0.0  |
|1.63 |2.232|1.183|2.142|0.0  |0.032|0.676|0.0  |0.69|2.282|0.307|
|0.0  |0.346|1.087|1.0  |1.923|0.0  |0.0  |0.0  |0.0 |0.0  |0.0  |


</details>


### Shard allocated

```python
shard_allocator(shards, nodes, algorithm_name):
```

<details>
 <summary>
Example of shards allocated with random algorithm
 </summary>

|shard|node |
|-----|-----|
|1    |2    |
|2    |2    |
|3    |1    |
|4    |1    |
|5    |2    |

</details>
