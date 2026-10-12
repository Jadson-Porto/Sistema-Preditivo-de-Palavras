# Sistema Preditivo de Palavras (Bayesian Next-Word Predictor)

Algoritmo de autocompletação e predição de texto baseado no **Teorema de Bayes** e estatística de **N-Grams** com **Laplace Smoothing**. 

Projeto desenvolvido como Exercício-Programa (EP) para a disciplina de **Análise Exploratória de Dados** (5º Semestre).

---

## Visão Geral

Diferente de abordagens baseadas em redes neurais profundas (LLMs ou RNNs), este projeto explora um modelo de linguagem probabilístico clássico. O objetivo principal é tornar transparente a inferência probabilística, demonstrando como a frequência relativa de sequências de palavras combinada com prioris estatísticas e técnicas de suavização determina a próxima palavra mais provável em um contexto.

### Destaques do Projeto
- **Zero Dependências Externas**: Implementado utilizando exclusivamente a biblioteca padrão do Python 3.10+.
- **Modelagem Bayesiana Explícita**: Separação clara entre *Prior*, *Verossimilhança (Likelihood)* e *Evidência (Normalização)*.
- **Tratamento de Contextos Inéditos**: Aplicação de *Laplace Smoothing* ($\alpha = 1.0$) para evitar o problema de probabilidade nula (zeroshot/OOV).
- **Cobertura de Testes**: Suíte de testes automatizados com 100% de aprovação (17/17 casos de unidade e integração).

---

## Como Rodar (Passo a Passo Rápido)

Siga os passos abaixo para testar o projeto rapidamente em sua máquina local:

### 1. Clonar o Repositório
Abra o terminal e baixe o código-fonte:
```bash
git clone https://github.com/seu-usuario/sistema-preditivo-de-palavras.git
cd sistema-preditivo-de-palavras
```

### 2. Verificar o Python
Certifique-se de que você tem o Python 3.10 ou superior instalado:
```bash
python --version
```

### 3. Testar uma Predição Simples
Execute o comando passando a frase que deseja prever a próxima palavra:
```bash
python -m src.main "O aluno estudou para a prova de"
```
> **Saída esperada:**
> ```text
> Sugestão: Estatística, Confiança: 71.58%
> ```

---

## Exemplos de Uso e Parâmetros

Você pode personalizar a execução usando as seguintes opções de linha de comando:

* **Obter um Ranking de Sugestões (Top-K):**
  ```bash
  python -m src.main "A análise exploratória de" -k 3
  ```
  > **Saída:**
  > ```text
  > Sugestão: Dados, Confiança: 81.35%
  > Sugestão: A, Confiança: 0.99%
  > Sugestão: De, Confiança: 0.95%
  > ```

* **Ajustar o Parâmetro de Suavização ($\alpha$):**
  ```bash
  python -m src.main "O teorema de" -a 0.5
  ```

* **Usar um Corpus Personalizado:**
  ```bash
  python -m src.main "Sua frase aqui" -c "caminho/para/seu_corpus.txt"
  ```

| Flag / Parâmetro | Descrição | Padrão |
| :--- | :--- | :--- |
| `frase` | Frase cujo próximo token será predito | `"O aluno estudou para a prova de"` |
| `-k`, `--top` | Quantidade de palavras sugeridas no ranking | `1` |
| `-a`, `--alpha` | Parâmetro de suavização de Laplace | `1.0` |
| `-c`, `--corpus` | Caminho para um arquivo de texto próprio | `data/corpus.txt` |

---

## Arquitetura do Projeto

O código é altamente modularizado para garantir facilidade de leitura e manutenção:

```text
sistema-preditivo-de-palavras/
├── data/
│   └── corpus.txt          # Corpus textual de treinamento do domínio
├── src/
│   ├── __init__.py         # Exportação da API do pacote
│   ├── __main__.py         # Ponto de entrada secundário para CLI
│   ├── config.py           # Tokens especiais e constantes do sistema
│   ├── preprocessing.py    # Sanitização, tokenização e estruturação em documentos
│   ├── ngrams.py           # Mapeamento e contagem de frequências (1-gram a N-gram)
│   ├── bayes.py            # Motor de inferência Bayesiana e cálculo de posteriors
│   └── main.py             # Interface de Linha de Comando (CLI)
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py # Testes da camada de limpeza de texto
│   ├── test_ngrams.py        # Testes da extração de N-grams
│   ├── test_bayes.py         # Testes dos cálculos de probabilidade
│   └── test_integration.py   # Testes de integração do sistema completo
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Executando os Testes Automatizados

O projeto utiliza o framework `unittest` nativo do Python. Para rodar a suíte completa de testes unitários e de integração:

```bash
python -m unittest discover -s tests
```

**Saída esperada:**
```text
................I
----------------------------------------------------------------------
Ran 17 tests in 0.012s

OK
```

Os testes cobrem:
- Sanitização e preservação de caracteres acentuados da língua portuguesa.
- Tratamento de palavras fora do vocabulário (OOV).
- Validação matemática (soma das probabilidades em $1.0$).
- Avaliação de acurácia com os casos de teste exigidos.

---

## Autor

Desenvolvido por **Jadson** como Exercício-Programa (EP) da disciplina de **Análise Exploratória de Dados** (5º Semestre).

- **Linguagem**: Python 3.10+.