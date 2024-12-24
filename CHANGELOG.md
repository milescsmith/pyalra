# CHANGELOG

## [1.6.2] - 2024/07/18

### Changed

- Downgrade required numpy version

## [1.6.1] - 2024/07/18

### Changed

- Updated `.gitignore`

### Fixed

- Removed a issue in pyproject.toml that caused the package to not actually install anything

## [1.6.0] - 2024/07/16

### Added

- This CHANGELOG

### Changed

- `choose_k()` now returns the randomized_svd results so that we do not needlessly recalculate the matrices
- No longer convert sparse matrices to dense.
- Replaced instances of slicing an array and setting the masked values with `np.copyto()`

## [1.5.0] - 2024/07/04

### Changed

- Switched from poetry to PDM, split ruff and mypy configurations into their own respective files

## [1.4.0] - 2024/02/05

### Changed

- Replaced use of `numpy.apply_along_axis` with the proper vectorized use of the same functions

## [1.3.0] - 2024/02/02

### Added

- Transparent support for processing sparce matrices
- Better logging capabilities

## [1.2.0] -2024/01/31

### Changed

- Actually return a tuple like the type hints state instead of a dict

### Fixed

- Omit `nan` values when performing stats
- Replace standard divide and sum functions with `np.divide` and `np.sum`, respectively, to handle the omitted `nan` 
    values

## [1.1.1] - 2024/01/31

### Fixed

- Another attempt to fix required numpy version

## [1.1.0] - 2024/01/31

### Fixed

- Required numpy version
- Removed typo in the `logger` submodule

## [1.0.0] - 2024/01/30

### Changed

- Everything fully implemented


[1.6.1]: https://github.com/milescsmith/pyalra/releases/compare/1.6.0..1.6.1
[1.6.0]: https://github.com/milescsmith/pyalra/releases/compare/1.5.0..1.6.0
[1.5.0]: https://github.com/milescsmith/pyalra/releases/compare/1.4.0..1.5.0
[1.4.0]: https://github.com/milescsmith/pyalra/releases/compare/1.3.0..1.4.0
[1.3.0]: https://github.com/milescsmith/pyalra/releases/compare/1.2.0..1.3.0
[1.2.0]: https://github.com/milescsmith/pyalra/releases/compare/1.1.1..1.2.0
[1.1.1]: https://github.com/milescsmith/pyalra/releases/compare/1.1.0..1.1.1
[1.1.0]: https://github.com/milescsmith/pyalra/releases/compare/1.0.1..1.1.0
[0.0.1]: https://github.com/milescsmith/pyalra/releases/tag/v1.0.0