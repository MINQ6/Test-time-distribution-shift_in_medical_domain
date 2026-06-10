"""Training subpackage.

서브모듈을 직접 import해서 사용하는 구조를 기본으로 두고,
__init__ 에서는 eager import를 하지 않는다.
이렇게 해야 training/clustering 같은 서브모듈 import 시
불필요한 순환 import와 heavy dependency 초기화를 줄일 수 있다.
"""

