#!/usr/bin/env python

sections = 4.0
transforms_per_section = 2.0
lost_per_layer = 2.0
initial_image_size = 316.0

print("Initial image size: %s" % initial_image_size)

print("Down convolutions")

current_matrix_size = initial_image_size - lost_per_layer * transforms_per_section
print("After down section 1: %f" % current_matrix_size)

for s in range(2, int(sections + 1)):
    current_matrix_size = current_matrix_size / 2 - lost_per_layer * transforms_per_section
    print("After down section %d: %f" % (s, current_matrix_size))

print("Up convolutions")

for s in range(1, int(sections + 1)):
    current_matrix_size = current_matrix_size * 2 - lost_per_layer * transforms_per_section
    print("After up section %d: %f" % (s, current_matrix_size))
