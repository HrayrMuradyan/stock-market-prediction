# An Automated Deep Learning Framework for Forecasting Stock Prices of Major Tech Companies

Author: Hrayr Muradyan

## About

Financial markets are complex, noisy, and driven by a mix of long-term fundamentals and short-term dynamics. Predicting stock price changes remains an unsolved problem, because they are often influenced by a wide range of measurable and unmeasurable factors — many of which interact in nonlinear and unpredictable ways. Unexpected executive decisions or sudden geopolitical events are examples of such variables.

These unobservable factors introduce noise and structural breaks that can invalidate any carefully designed statistical model. Thus, the goal of this project is not to "solve" stock market prediction problem, but to build a systematic framework that learns from patterns where they exist — while understanding the market's randomness.

## Dependencies

The project is done using the following external dependencies:

- [Docker](https://www.docker.com/): To design reproducible, isolated environments for development and deployment.
- [Git](https://git-scm.com/): For version control and collaboration on code.

Internal dependencies include:
- [Python](https://www.python.org/): Core programming language used throughout the project.
- [Apache Airflow](https://airflow.apache.org/): Workflow orchestration tool used to define and schedule DAGs.
- [PostgreSQL](https://www.postgresql.org/): Backend metadata database for Airflow.

Required Python packages are separated into **Production** ([requirements.txt](./requirements.txt)) and **Development** ([requirements-dev.txt](./requirements-dev.txt)) files to separate runtime requirements from tools used for testing and local development. Visit the respective files for more information.


## Usage

## License

This repository is licensed under the MIT License. For more information, please see the [LICENSE](./LICENSE) file.

## Contributing

Any contributions are welcomed — whether you have found a bug, have a suggestion, or simply want to share your expertise or opinion. Please read the [contributing guidelines](CONTRIBUTING.md) to get started.




