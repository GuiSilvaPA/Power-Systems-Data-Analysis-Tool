class EventData():

    def __init__(self, time):

        self.events   = []
        self.names    = []
        self.n_events = 0
        self.time     = time

    def new_event(self, name, evento, info1, param1, time, info2=0, info3=0, param2=0, param3=0):

        self.names.append(name)
        self.events.append([f'{evento:3d}    {info1:4d} {info2:4} {info3:4}      {param1:7.3f}  {param2:7.3f}     {time:5.3f}  \"xxxxxxxxxx  \"  \"xxxxxxxxxx  \"  {param3:7.3f} /\n'])
        print(f'Event number {len(self.events)} created')



    def append(self, n_event, evento, info1,  param1, time, info2=0, info3=0, param2=0, param3=0):

        self.events[n_event-1].append(f'{evento:3d}    {info1:4d} {info2:4} {info3:4}      {param1:7.3f}  {param2:7.3f}     {time:5.3f}  \"xxxxxxxxxx  \"  \"xxxxxxxxxx  \"  {param3:7.3f} /\n')

    def save(self, save_path):

      file = [f'{self.time: 7.3f}/\n']

      for i in range(len(self.names)):

          file.append(f'{i+1:3d} \'{self.names[i]: <45}\' /\n')

          for evt in self.events[i]:
              file.append(evt)
              

          file.append(' -99/ \n')

      file.append(' -999/ \n')

      with open(save_path, 'w') as f:
            for line in file:
                f.write(line)