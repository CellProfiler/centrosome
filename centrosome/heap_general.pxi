cdef extern from "stdlib.h":
   ctypedef unsigned long size_t
   void free(void *ptr)
   void *malloc(size_t size)
   void *realloc(void *ptr, size_t size)

cdef struct Heap:
    unsigned int items
    unsigned int space
    Heapitem *data
    Heapitem **ptrs

cdef inline Heap *heap_from_numpy2():
    cdef unsigned int k
    cdef Heap *heap 
    heap = <Heap *> malloc(sizeof (Heap))
    heap.items = 0
    heap.space = 1000
    heap.data = <Heapitem *> malloc(heap.space * sizeof(Heapitem))
    heap.ptrs = <Heapitem **> malloc(heap.space * sizeof(Heapitem *))
    for k in range(heap.space):
        heap.ptrs[k] = heap.data + k
    return heap

cdef inline void heap_done(Heap *heap):
   free(heap.data)
   free(heap.ptrs)
   free(heap)

cdef inline void swap(unsigned int a, unsigned int b, Heap *h):
    h.ptrs[a], h.ptrs[b] = h.ptrs[b], h.ptrs[a]


######################################################
# heappop - inlined
#
# pop an element off the heap, maintaining heap invariant
# 
# Note: heap ordering is the same as python heapq, i.e., smallest first.
######################################################
cdef inline void heappop(Heap *heap,
                  Heapitem *dest):
    cdef unsigned int i, smallest, l, r # heap indices
    
    #
    # Start by copying the first element to the destination
    #
    dest[0] = heap.ptrs[0][0]
    heap.items -= 1

    # if the heap is now empty, we can return, no need to fix heap.
    if heap.items == 0:
        return

    #
    # Move the last element in the heap to the first.
    #
    swap(0, heap.items, heap)

    #
    # Restore the heap invariant.
    #
    i = 0
    smallest = i
    while True:
        # loop invariant here: smallest == i
        
        # find smallest of (i, l, r), and swap it to i's position if necessary
        l = i*2+1 #__left(i)
        r = i*2+2 #__right(i)
        if l < heap.items:
            if smaller(heap.ptrs[l], heap.ptrs[i]):
                smallest = l
            if r < heap.items and smaller(heap.ptrs[r], heap.ptrs[smallest]):
                smallest = r
        else:
            # this is unnecessary, but trims 0.04 out of 0.85 seconds...
            break
        # the element at i is smaller than either of its children, heap invariant restored.
        if smallest == i:
                break
        # swap
        swap(i, smallest, heap)
        i = smallest
        
##################################################
# heappush - inlined
#
# push the element onto the heap, maintaining the heap invariant
#
# Note: heap ordering is the same as python heapq, i.e., smallest first.
##################################################
cdef inline void heappush(Heap *heap,
                          Heapitem *new_elem):
  cdef unsigned int child         = heap.items
  cdef unsigned int parent
  cdef unsigned int k
  cdef Heapitem *new_data

  # grow if necessary
  if heap.items == heap.space:
      heap.space = heap.space * 2
      new_data = <Heapitem *> realloc(<void *> heap.data, <size_t> (heap.space * sizeof(Heapitem)))
      heap.ptrs = <Heapitem **> realloc(<void *> heap.ptrs, <size_t> (heap.space * sizeof(Heapitem *)))
      for k in range(heap.items):
          heap.ptrs[k] = new_data + (heap.ptrs[k] - heap.data)
      for k in range(heap.items, heap.space):
          heap.ptrs[k] = new_data + k
      heap.data = new_data

  # insert new data at child
  heap.ptrs[child][0] = new_elem[0]
  heap.items += 1

  # restore heap invariant, all parents <= children
  while child>0:
      parent = (child + 1) // 2 - 1 # __parent(i)
      
      if smaller(heap.ptrs[child], heap.ptrs[parent]):
          swap(parent, child, heap)
          child = parent
      else:
          break
