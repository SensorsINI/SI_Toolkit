from tqdm import tqdm


def filter_datasets(dfs, args):
    data_filter = DataFilter(args)
    dfs_filtered = []
    for df in tqdm(dfs, desc="Applying filters"):
        df_filtered_list = data_filter.apply_filters(df)
        for df_filtered in df_filtered_list:
            if len(df_filtered) == 0:
                continue
            dfs_filtered.append(df_filtered)

    return dfs_filtered


class DataFilter:
    def __init__(self, args):
        # Check if 'filters' field exists in the configuration
        if hasattr(args, 'filters') and isinstance(args.filters, list):
            self.filter_funcs = []
            for data_filter in args.filters:
                column = data_filter['column']
                condition = data_filter['condition']
                operator, value_str = condition.split(" ", 1)
                try:
                    value = float(value_str)
                except ValueError:
                    raise ValueError(f"Invalid value in condition: {value_str}")
                use_absolute = data_filter.get('absolute', False)

                # Function to apply or bypass abs()
                def apply_abs_if_needed(dataset_values, use_abs):
                    return dataset_values.abs() if use_abs else dataset_values

                # Creating lambda functions based on the operator
                if operator == '<':
                    self.filter_funcs.append(
                        lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) < v])
                elif operator == '<=':
                    self.filter_funcs.append(
                        lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) <= v])
                elif operator == '>':
                    self.filter_funcs.append(
                        lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) > v])
                elif operator == '>=':
                    self.filter_funcs.append(
                        lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) >= v])
                elif operator == '==':
                    self.filter_funcs.append(
                        lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) == v])
                elif operator == '!=':
                    self.filter_funcs.append(
                        lambda df, c=column, v=value: df[apply_abs_if_needed(df[c], use_absolute) != v])
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
        else:
            # If 'filters' field is not in config, setup to bypass filtering
            self.filter_funcs = [lambda df: df]

    def apply_filters(self, df):
        for func in self.filter_funcs:
            df = func(df)

        if df.empty:
            return []
        dataframes = self.split_on_discontinuities(df)
        return dataframes

    def split_on_discontinuities(self, df):
        """Split the DataFrame where there are discontinuities in the index."""

        discontinuities = df.index.to_series().diff().fillna(0).abs() > 1
        split_indices = discontinuities[discontinuities].index

        # Split the dataframe based on the discontinuity indices
        dataframes = []
        previous_index = df.index[0]
        for idx in split_indices:
            split_df = df.loc[previous_index:idx - 1]
            if not split_df.empty:
                dataframes.append(split_df)
            previous_index = idx

        # Add the final remaining segment
        remaining_df = df.loc[previous_index:]
        if not remaining_df.empty:
            dataframes.append(remaining_df)

        return dataframes