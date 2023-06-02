
file_names = []

experiment = 3

if experiment == 1:

    for i in range(1, 401, 50):
        file_name = f"SGP_10_{i}_10_1"
        file_names.append(file_name)

    print(file_names)

if experiment == 2:

    for i in range(2, 101, 5):
        file_name = f"SGP_{i}_400_10_1"
        file_names.append(file_name)

    print(file_names)

if experiment == 3:

    for i in range(1, 101, 20):
        file_name = f"SGP_10_400_{i}_1"
        file_names.append(file_name)
        file_name = f"SGP_10_400_{i}_2"
        file_names.append(file_name)
        file_name = f"SGP_10_400_{i}_3"
        file_names.append(file_name)

    print(file_names)


