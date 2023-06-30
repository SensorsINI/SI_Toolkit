
file_names = []

experiment = 1

if experiment == 1:

    for i in range(50, 401, 25):
        file_name = f"SGP_15_{i}_25_1"
        file_names.append(file_name)
        file_name = f"SGP_15_{i}_25_2"
        file_names.append(file_name)
        file_name = f"SGP_15_{i}_25_3"
        file_names.append(file_name)

    print(file_names)

if experiment == 2:

    for i in range(26, 27, 1):
        file_name = f"SGP_{i}_400_25_1"
        file_names.append(file_name)
        file_name = f"SGP_{i}_400_25_2"
        file_names.append(file_name)
        file_name = f"SGP_{i}_400_25_3"
        file_names.append(file_name)
        file_name = f"SGP_{i}_400_25_4"
        file_names.append(file_name)
        file_name = f"SGP_{i}_400_25_5"
        file_names.append(file_name)

    print(file_names)

if experiment == 3:

    for i in range(25, 101, 10):
        file_name = f"SGP_100_400_{i}_1"
        file_names.append(file_name)
        file_name = f"SGP_100_400_{i}_2"
        file_names.append(file_name)
        file_name = f"SGP_100_400_{i}_3"
        file_names.append(file_name)
        file_name = f"SGP_100_400_{i}_4"
        file_names.append(file_name)
        file_name = f"SGP_100_400_{i}_5"
        file_names.append(file_name)

    print(file_names)


