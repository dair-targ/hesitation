import optparse


class Action(object):
  def __init__(self):
    self._verbose = False
  
  def build_parser(self):
    parser = optparse.OptionParser('%prog [options] INPUT OUTPUT')
    parser.add_option(
      '-v', '--verbose',
      dest='verbose',
      action='store_true',
      default=False,
      help='Write what`s going on',
    )
    return parser
  
  def build_parameters(self, options, args, on_error):
    return dict()
  
  def run(self, on_error):
    pass
  
  def main(self):
    parser = self.build_parser()
    options, args = parser.parse_args()
    parameters = self.build_parameters(options, args, parser.error)
    self.run(on_error=parser.error)
