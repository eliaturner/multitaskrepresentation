from data.data_generator import *
from data.custom_data_generator import *

#
#
from data.functions import *

vmax = 3

single_single_args = {'input_type':'transient',
                      'output_type':'single',
                      'steps':300,
                      'transient_amplitude': 1
                      }

single_single_tonic_args = {'input_type':'tonic',
                      'output_type':'single',
                      'steps':300,
                      'transient_amplitude': 1
                      }

single_multi_args = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':100,
                     'extra':0,
                     'n_tasks_total': None
                      }

single_multi_tonic_args = {'input_type':'tonic',
                      'output_type':'multiple',
                      'steps':100,
                     'extra':0,
                     'n_tasks_total': None
                      }


multi_single_args = {'input_type':'multi',
                      'output_type':'single',
                      'steps':400
                      }

multi_multi_args = {'input_type':'multi',
                      'output_type':'multiple',
                      'steps':300
                      }

single_multi_args_7 = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':300,
                       'n_tasks_total': 7
                      }

multi_multi_args_7 = {'input_type':'multi',
                      'output_type':'multiple',
                      'steps':300,
                      'n_tasks_total': 7
                      }

single_multi_args_3 = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':300,
                       'n_tasks_total': 3
                      }

single_multi_args_4 = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':300,
                       'n_tasks_total': 4
                      }

single_multi_args_10 = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':300,
                       'n_tasks_total': 10
                      }


single_multi_args_6 = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':300,
                       'n_tasks_total': 6
                      }


single_multi_args_5 = {'input_type':'transient',
                      'output_type':'multiple',
                      'steps':300,
                       'n_tasks_total': 5
                      }

multi_multi_args_3 = {'input_type':'multi',
                      'output_type':'multiple',
                      'steps':300,
                      'n_tasks_total': 3
                      }


multi_all_args = {'input_type':'multi',
                      'output_type':'all',
                      'steps':300
                      }
