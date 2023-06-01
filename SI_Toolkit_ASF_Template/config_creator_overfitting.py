
file_names = []

experiment = 1

if experiment == 1:

    for i in range(1, 401, 50):
        file_name = f"SGP_10_{i}_10"
        file_names.append(file_name)

    print(file_names)

if experiment == 2:

    for i in range(2, 101, 5):
        file_name = f"SGP_{i}_400_10"
        file_names.append(file_name)

    print(file_names)



