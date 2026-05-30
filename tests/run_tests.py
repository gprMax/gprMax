# simple test runner, avoids pytest dependency
import sys
import os
# ensure repository root is on path so we can import the test modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib.util
# load our test modules directly by filename to bypass package import
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

def load_by_path(path):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# import the test modules
tf = load_by_path(os.path.join(root, 'tests', 'test_fractals.py'))
tm = load_by_path(os.path.join(root, 'tests', 'test_mixingmodels.py'))

if __name__ == '__main__':
    tf.test_bin_fractal_values_range()
    tf.test_bin_fractal_values_edgecases()
    tm.test_peplinski_materials_added()
    print('✅ all custom tests passed')
