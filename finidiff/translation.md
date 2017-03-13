* postpad(x, l > len(x)) -> x.resize(l), except that the latter's
  in-line and returns None so if we want something more functional, we
  implement it using numpy.concatenate

