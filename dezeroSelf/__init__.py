is_simple_core = False
if is_simple_core:
    from dezeroSelf.core_simple import Variable
    from dezeroSelf.core_simple import Function
    from dezeroSelf.core_simple import using_config
    from dezeroSelf.core_simple import no_grad
    from dezeroSelf.core_simple import as_array
    from dezeroSelf.core_simple import as_variable
    from dezeroSelf.core_simple import setup_variable
else:
    from dezeroSelf.core import Variable
    from dezeroSelf.core import Function
    from dezeroSelf.core import using_config
    from dezeroSelf.core import no_grad
    from dezeroSelf.core import as_array
    from dezeroSelf.core import as_variable
    from dezeroSelf.core import setup_variable  

setup_variable()