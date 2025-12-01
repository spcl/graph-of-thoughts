import random

# Step 1: generate two random sorted lists
def random_sorted_list(n, low=0, high=9):
    lst = [random.randint(low, high) for _ in range(n)]
    lst.sort()
    return lst

a = random_sorted_list(64)
b = random_sorted_list(64)

print("List A:", a)
print("List B:", b)

# Step 2: merge and then sort
merged = a + b
merged.sort()

print("Merged & Sorted:", merged)
