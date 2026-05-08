# slcp-offsetting

## Usage

Install [pixi](https://pixi.sh), then:

```bash
pixi install
```

To install a specific environment (e.g. `lint`, `test`, `coiled-deploy`):

```bash
pixi install -e lint
```

### Adding packages

From conda-forge (preferred):

```bash
pixi add <package>
```

From PyPI:

```bash
pixi add --pypi <package>
```

To add to a specific environment:

```bash
pixi add -e lint <package>
```

### JupyterLab

```bash
pixi run jupyter lab
```

### Linting

```bash
pixi run -e lint prek run --all-files
```

> [!IMPORTANT]
> This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org)
