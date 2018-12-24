import time

s = set({})
l = []
start = time.time()
for i in range(100000):
    s.add(i)
print(time.time()-start)
start = time.time()
for i in range(100000):
    l.append(i)
print(time.time()-start)
start = time.time()
for i in range(5000):
    s.pop()
print(time.time()-start)
start = time.time()
for i in range(5000):
    l.pop(4999)
print(time.time()-start)
