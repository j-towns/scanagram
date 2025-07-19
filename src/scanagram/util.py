###############################################################################
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copied from jax/_src/util.py because jax.util.safe_map and safe_zip are
# deprecated
def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))

def safe_zip(*args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(zip(*args))
###############################################################################

def all_equal(xs):
    xs = list(xs)
    if len(xs) == 0:
        return True
    else:
        x, *xs = xs
        return all(y == x for y in xs)

def unzip_scanvars(scanvars):
    argnums = []
    axes = []
    prefills = []
    for n, s in scanvars:
        argnums.append(n)
        axes.append(s.axis)
        prefills.append(s.prefill)
    return tuple(argnums), tuple(axes), tuple(prefills)
