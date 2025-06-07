import numpy as np
import pandas as pd


def process_timestamps(df, keep_timestamp=False):
    # Convert timestamp to datetime if it isn't already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract useful temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['seconds_since_epoch'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # Drop original timestamp
    if not keep_timestamp:
        df.drop('timestamp', axis=1, inplace=True)
    return df


def create_sequences(data: pd.DataFrame, seq_length: int):
    sequences = []
    sequence_labels = []
    data_size = len(data) - (len(data) % seq_length)

    data_labels = data['label'].values
    data = data.drop(columns=['label'])

    for i in range(0, data_size, seq_length):
        # label = data_labels[i + seq_length - 1]  # Using last packet's label
        label = np.max(data_labels[i:i + seq_length])  # Using max of the sequence
        seq = data.iloc[i:i + seq_length, :].values
        sequences.append(seq)
        sequence_labels.append(label)

    return np.array(sequences), np.array(sequence_labels)


def create_sequences_based_on_session(data: pd.DataFrame):
    sequences = []
    sequence_labels = []

    data_labels = data['label'].values
    data = data.drop(columns=['label'])
    data["session_id"] = 0

    #session_id = 1
    item_index = 0
    session_sequence_items = []
    current_session_label_counts = {val: 0 for val in np.unique(data_labels)}
    current_session_name = ""  # in the form of "<source_ip>:<source_port>-<target_ip>:<target_port>"
    for row in data.itertuples(index=False):
        # create the source section of the session
        source_ip = ""
        for i in range(1, 9):
            source_ip += str(getattr(row, f"ip_source_part{i}")) + "."
        source_ip = source_ip[:-1]  # exclude the last dot
        source_port = str(getattr(row, "source_port_label_normalized"))

        # create the destination section of the session
        destination_ip = ""
        for i in range(1, 9):
            destination_ip += str(getattr(row, f"ip_destination_part{i}")) + "."
        destination_ip = destination_ip[:-1] # exclude the last dot
        destination_port = str(getattr(row, "destination_port_label_normalized"))

        session_name_for_the_item = f"{source_ip}:{source_port}-{destination_ip}:{destination_port}"
        if current_session_name == "":
            # set the session name for the first session
            current_session_name = session_name_for_the_item

        if session_name_for_the_item != current_session_name:
            # a new session starts, record the previous session
            session_label = max(current_session_label_counts, key=current_session_label_counts.get)
            sequences.append(session_sequence_items[:])
            sequence_labels.append(session_label)

            # cleanup for the next session
            session_sequence_items = []
            #session_id += 1
            current_session_label_counts = {val: 0 for val in np.unique(data_labels)}

        #data[item_index]["session_id"] = session_id
        session_sequence_items.append(row)
        current_session_label_counts[data_labels[item_index]] += 1
        item_index += 1

    return np.array(sequences), np.array(sequence_labels)


def create_time_window_sequences(
    data: pd.DataFrame,
    window_seconds: int = 5,
    label_reducer=max            # or np.max / majority vote, etc.
):
    """
    Split traffic into variable-length windows whose **wall-clock span**
    is <window_seconds>.  Returns two Python lists:
        • sequences : List[np.ndarray]   shape (seq_len, n_features)
        • labels    : List[int]          one label per sequence
    """

    # 1. Ensure chronological order
    data = data.sort_values("timestamp").reset_index(drop=True)

    # 2. Convert the 'timestamp' column to epoch seconds **once**
    #    (works whether the column is already pd.Timestamp or a string)
    epoch_ts = pd.to_datetime(data["timestamp"]).astype("int64") / 1e9
    data = data.assign(__epoch_ts=epoch_ts)

    all_seqs, all_labels = [], []
    cur_start_ts = None
    cur_seq, cur_labels = [], []

    for _, row in data.iterrows():
        ts      = row["__epoch_ts"]          # float64 seconds
        label   = row["label"]

        # drop label & both timestamp columns from the feature tensor
        feats = row.drop(["label", "timestamp", "__epoch_ts"]).to_numpy(dtype=np.float32)

        # open new window if needed
        if cur_start_ts is None:
            cur_start_ts = ts

        if ts - cur_start_ts < window_seconds:
            # same window
            cur_seq.append(feats)
            cur_labels.append(label)
        else:
            # flush previous window
            all_seqs.append(np.stack(cur_seq))
            all_labels.append(label_reducer(cur_labels))

            # start a new window
            cur_start_ts = ts
            cur_seq      = [feats]
            cur_labels   = [label]

    # flush final window
    if cur_seq:
        all_seqs.append(np.stack(cur_seq))
        all_labels.append(label_reducer(cur_labels))

    return all_seqs, all_labels
