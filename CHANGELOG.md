# Changelog

All notable changes to this project will be documented in this file.


## [0.5.1] - 2026-10-26


### Documentation

- synchronize README examples, add docstring (#24) ([a942c96](https://github.com/OWNER/REPO/commit/a942c96d547c6f69f2c0e3005070cad20e74d85b))



### Fixed

- critical bugs and fortran order (#22) ([8b77f96](https://github.com/OWNER/REPO/commit/8b77f96f0ffeb138287b107cb2c1b2fb5e803be0))


- strengthen model validation, prevent aliasing, and refine grid utilities (#23) ([12e36a0](https://github.com/OWNER/REPO/commit/12e36a09f142025b66874cee6a639325cd09fde7))



## [0.5.0] - 2026-10-17


### Added

- geological model extensions (#20) ([61bdc6f](https://github.com/OWNER/REPO/commit/61bdc6fa80fe30eb5e946f7b177ae53f837fd3a1))


- grid manipulation utilities (#21) ([5f823be](https://github.com/OWNER/REPO/commit/5f823be0c123fd2ab6e65e7e24b3113e75b1f81e))



### Changed

- homogenize parameters (#18) ([96a0b01](https://github.com/OWNER/REPO/commit/96a0b017e551800227a51afdc9d5dc64035d8488))



### Documentation

- complete english docstrings (#19) ([5a07341](https://github.com/OWNER/REPO/commit/5a07341f06f4458959a94cbe3d2d03eb47ac4ff9))



### Maintenance

- seiscode-models-0.5.0 ([d4db625](https://github.com/OWNER/REPO/commit/d4db62541a2b2b4c29c881fa1fe4d43b6c680b23))



## [0.4.0] - 2026-10-16


### Added

- add elliptical inclusion primitive (#17) ([13b03e5](https://github.com/OWNER/REPO/commit/13b03e52da27208da402c5368f4df1689b55dbfe))



### Documentation

- rewrite README with accurate API examples (#16) ([da118d9](https://github.com/OWNER/REPO/commit/da118d9b6e9682f3159795a87c4bd8eaaddd795a))



### Maintenance

- switch to manual workflow dispatch with automated version calculation (#15) ([63eaab1](https://github.com/OWNER/REPO/commit/63eaab1964bdbe501d551e20daca1b37eadf7df5))


- seiscode-models-0.4.0 ([d2b7c90](https://github.com/OWNER/REPO/commit/d2b7c90cac97a669b5186e32620eb531e18fb9aa))



## [0.3.1] - 2026-10-15


### Documentation

- improve read me (#12) ([a0f643b](https://github.com/OWNER/REPO/commit/a0f643bb2ebbfb0bbc95b733ca27732c5236abb2))



### Maintenance

- improve CI workflow (#10) ([5419187](https://github.com/OWNER/REPO/commit/54191873466774a197e3edf22c9239b885afe525))


- improve continuous integration by using cache for pip (#11) ([3f3d125](https://github.com/OWNER/REPO/commit/3f3d125c659ae563d8112c2813c96afa3f9babcc))


- improve and fix release and publish workflow (#13) ([e86bd3e](https://github.com/OWNER/REPO/commit/e86bd3e0a02779476670789657912e9fac649f42))


- seiscode-models-0.3.1 ([bd700fd](https://github.com/OWNER/REPO/commit/bd700fd99c73cb06c64fa33fc9e413901736ef3b))


- manage git cliff configuration (#14) ([d178fa7](https://github.com/OWNER/REPO/commit/d178fa72fab874b3609c08207ec9615d6d8ffc6e))


- seiscode-models-0.3.1 ([19fd6a3](https://github.com/OWNER/REPO/commit/19fd6a38dc3fbb236e6726367a5959d6dc245c54))



## [0.3.0] - 2026-10-13


### Added

- improve geological functions ([f10c277](https://github.com/OWNER/REPO/commit/f10c277736b2ac2808548b4bedf4e315c2ebe7e9))


- add new methods and run ruff formating ([2ceab18](https://github.com/OWNER/REPO/commit/2ceab18cf2f4455bbc474d58b5f39fe2a3d155ff))



### Fixed

- version comes from pyproject.toml only ([5bdfc6f](https://github.com/OWNER/REPO/commit/5bdfc6fc54e8a8de5ade5b32fbb4cf2f7985be76))


- Add .npy if missing and manage C and F ordering ([b622de2](https://github.com/OWNER/REPO/commit/b622de210b2c3812871d771bac03bc973a9c9b53))



### Release

- prepare for 0.3.0 ([1b4160b](https://github.com/OWNER/REPO/commit/1b4160bddd91bf353aad8be8242de86f0b5a238f))


- prepare for 0.3.0 ([20ee83d](https://github.com/OWNER/REPO/commit/20ee83d4a5d0e133277ef93eace0be609a48ba59))



### Style

- run ruff check --fix ([77ffe49](https://github.com/OWNER/REPO/commit/77ffe49cef89632d33df7a6a2087846c4122cb89))



## [0.2.0] - 2026-10-12


### Added

- add VelocityModel class ([c814510](https://github.com/OWNER/REPO/commit/c8145102608d5d7324c2e7a533cc53747fc1a450))



### Documentation

- update changelog ([dfb52e1](https://github.com/OWNER/REPO/commit/dfb52e107a5b1e961d1db17969782cbcbbeb56c7))


- update changelog ([285180a](https://github.com/OWNER/REPO/commit/285180aebb529b956a48f6c48d5fc78de02c1270))



### Fixed

- change version for pyproject.toml ([c09f4a6](https://github.com/OWNER/REPO/commit/c09f4a65a66f8b84212225922f1151237828fe12))



## [0.1.0] - 2026-10-12


### Added

- setup package structure and dtype utility for f32/f64 precision ([244f9ae](https://github.com/OWNER/REPO/commit/244f9ae9de2368347f3268f25ffee59b786c0b77))


- implement core velocity primitives and validation with f32/f64 support ([81c37b6](https://github.com/OWNER/REPO/commit/81c37b6c4b21ad2db5b592a67cd5a3414304e2bb))


- implement geological velocity models with precision support ([da0d2c4](https://github.com/OWNER/REPO/commit/da0d2c428fa6ec6bd74782bd07540f4203cfc9c2))


- add model export functions for seismic solvers ([a8c4ef6](https://github.com/OWNER/REPO/commit/a8c4ef6ee05b21f00fd5ecd47d716ee34d273d97))


- add visualization module ([799ed34](https://github.com/OWNER/REPO/commit/799ed34e3dbc2e86e7d9c23618c9cee7be5eafc4))



### Documentation

- update CHANGELOG ([0ea5b2c](https://github.com/OWNER/REPO/commit/0ea5b2c2688e3cf00a53373aa8015a2711cadb27))


- Core velocity primitives and validation with f32/f64 support ([8a50266](https://github.com/OWNER/REPO/commit/8a5026629bdbd5e5bcb8220d923c27a8184f105c))


- update changelog ([5e9361a](https://github.com/OWNER/REPO/commit/5e9361a61154daba14d47441cd19bb3824aa54a0))



### Fixed

- ensure dtype annotations are compatible with Python 3.8 ([011a5d0](https://github.com/OWNER/REPO/commit/011a5d0362e395fc3c0a1ed6ecba3f94cf4fd077))


- restore Python 3.9 compatibility in IO annotations ([57180f7](https://github.com/OWNER/REPO/commit/57180f74ccea3208d240e5abc7725a7a2931a7e9))


- restore Python 3.8/3.9 compatibility in IO annotations ([b141ab6](https://github.com/OWNER/REPO/commit/b141ab695436b903c300ce23ddeb10fa5b11a872))


- ensure type annotations (tuple)  support Python 3.8 ([9385144](https://github.com/OWNER/REPO/commit/938514468a6a5ac9bee7ae375da9913fb9eb2542))



### Maintenance

- add ruff configuration and changelog ([82404f7](https://github.com/OWNER/REPO/commit/82404f762f15f0b81409b29c7ad5f2b06ce8f364))


- update release and publish ([051034f](https://github.com/OWNER/REPO/commit/051034f939618eeae091df6b25a4475d05fac792))


