import os
import event_stream
import json
import subprocess

import numpy as np
from metavision_core.event_io import EventsIterator


def import_data(file_path,sample_size,start_event):
    
    file_extension = os.path.splitext(file_path)[1]
    filename = os.path.basename(file_path)

    if file_extension == '.es':
            #get the starting time of the data
        command = f"C:\\command_line_tools\\build\\release\\statistics {file_path}"
        result = subprocess.run(["powershell.exe","-command",command], capture_output=True, text=True)
        # Check if the PowerShell command executed successfully
        if result.returncode == 0:
            # Parse the output as JSON
            output_json = json.loads(result.stdout)

            # Extract the value of begin_representation
            begin_representation = output_json["begin_representation"]
            begin_representation = begin_representation[:8]
            print('t value: ',begin_representation)
            # Decoder's only argument is an Event Stream file path
        decoder = event_stream.Decoder(file_path)
        if decoder.type == 'generic':
            print('generic events')
        else:
            print(f'{decoder.type} events, {decoder.width} x {decoder.height} sensor')

        # Initiate an empty list to store all chunks
        all_chunks = []

        # chunk is a numpy array whose dtype depends on the decoder type
        for chunk in decoder:
        # print('{} events, ts = [{} µs, {} µs]'.format(len(chunk), chunk['t'][0], chunk['t'][-1]))
            all_chunks.append(chunk)

        # Concatenate all chunks into a single numpy array
        all_events = np.concatenate(all_chunks)

        # Create a new dtype with the same fields, but changing 'on' to 'p'
        new_dtype = [('t', 'float64'), ('x', '<u2'), ('y', '<u2'), ('p', 'int')]


        # Create a new array with the new dtype, and convert the values
        data_ingest = np.empty(all_events.shape, dtype=new_dtype)
        for name in all_events.dtype.names:
            if name == 'on':
                data_ingest['p'] = all_events['on'].astype(int)
            else:
                data_ingest[name] = all_events[name]

        print('begin: ',data_ingest['t'][0])
        print('end: ',data_ingest['t'][-1])
        for events in range(len(data_ingest)):
            data_ingest[events]['t'] -= float(begin_representation)
        data_ingest = data_ingest[data_ingest['t'] <= (sample_size + start_event)]
        print('after trim array start: ', data_ingest['t'][0])
        print('after trim array length: ', data_ingest['t'][-1])
        return data_ingest
    
    elif file_extension == '.raw':
                # Create an EventsIterator to read the events from the .raw file
        mv_iterator = EventsIterator(input_path=file_path)

        # Initiate empty lists to store the event data
        timestamps = []
        x_values = []
        y_values = []
        polarity_values = []

        # Iterate over the events and store the data in the lists
        for events in mv_iterator:
            for event in events:
                timestamps.append(event['t'])
                x_values.append(event['x'])
                y_values.append(event['y'])
                polarity_values.append(event['p'])

        # Create a numpy array with the same dtype as data_ingest
        dtype = [('t', 'float64'), ('x', '<u2'), ('y', '<u2'), ('p', 'int')]
        data_ingest = np.array(list(zip(timestamps, x_values, y_values, polarity_values)), dtype=dtype)
        print('begin: ',data_ingest['t'][0])
        print('end: ',data_ingest['t'][-1])
        for events in range(len(data_ingest)):
            data_ingest[events]['t'] -= float(data_ingest['t'][0])
        data_ingest = data_ingest[data_ingest['t'] <= (sample_size + start_event)]
        print('after trim array start: ', data_ingest['t'][0])
        print('after trim array length: ', data_ingest['t'][-1])
        return data_ingest

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")