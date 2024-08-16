### kleindl 

SMALL DESCRIPTION

#### goals
- run open source models
- proper acceleration support (ane, cuda)
- einops style tensor permutations
- benchmarking suite
- visualization suite
- focus on minimal line count

##### Outline (tmp)
What I want this project to look like and where I want it to go.
1. A working tensor and autograd library. This has already been done, but is being 
  refactored a bit. With this I have a base for all things that need to be calculated in a neural network.
2. Work on getting accelerators working in the library such as apple neural engine and cuda. This is to be
  done as low level as possible. Goal is to make everything low overhead, speedy, efficient, and 
  compatiable.
3. Get open source models such as llama working with this framework. Having the ability to locally run llm's
  is the future and we need more ways to do so. It needs to be accessible to everyone.
4. Compete with companies like tinycorp.