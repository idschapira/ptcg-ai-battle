# **Pokemon TCG AI Battle Challenge \- Estrutura e Planejamento**

## **Visão Geral do Projeto**

Este documento define a estrutura inicial para a competição Kaggle "Pokémon TCG AI Battle Challenge", servindo como base para organização no Cowork. O objetivo é treinar um agente de Inteligência Artificial para jogar Pokémon TCG contra outros agentes, utilizando o motor de simulação oficial fornecido pelo Kaggle e pela The Pokémon Company.

## **1\. Setup Inicial e Base de Dados**

A base de dados oficial da competição foi localizada no Kaggle e inclui os seguintes recursos essenciais que devem ser importados para o nosso ambiente:

* **Motor de Jogo (Game Engine):** ptcg\_engine.zip (Código-fonte oficial do simulador de batalha). *Nota crucial: Este motor deve ser usado estritamente para teste, verificação e treinamento local.*  
* **Metadados das Cartas (Inglês):** EN Card Data.csv (Contém ID, Nome, HP, Tipos, Ataques, etc.) e Card\_ID\_List\_EN.pdf (Referência visual).  
* **Metadados das Cartas (Japonês):** JP Card Data.csv e Card\_ID\_List\_JP.pdf (Opcional, mesma estrutura do inglês).  
* **Replays:** Acesso a replays diários dos melhores episódios para análise e possível Imitation Learning (IL).

## **2\. RoadMap do Projeto (Até Agosto/2026)**

O prazo final para submissão na categoria de Simulação é **16 de Agosto de 2026**. O planejamento seguirá as seguintes fases:

| Fase | Período Sugerido | Foco Principal | Entregável |
| :---- | :---- | :---- | :---- |
| Fase 1: Fundações e Setup | Semanas 1-2 | Infraestrutura, ingestão de dados (CSV) e integração do motor ptcg\_engine. | Ambiente rodando testes locais com o motor oficial. |
| Fase 2: Agentes Baseline | Semanas 3-4 | Criação de agentes baseados em regras/heurísticas (ex: escolhas aleatórias ou lógica condicional simples) para testar submissões. | Primeira submissão validada no Kaggle (submission.tar.gz com main.py e deck.csv). |
| Fase 3: Treinamento RL / DL | Semanas 5-7 | Implementação de Reinforcement Learning (PPO, DQN) usando os dados do simulador e análise de replays. | Agente capaz de aprender estratégias básicas e aumentar sua classificação (Rating μ). |
| Fase 4: Otimização e Submissão Final | Semanas 8-9 (Até 16/Ago) | Ajuste fino de hiperparâmetros, seleção rigorosa do deck e testes contra meta-decks. | Submissão das versões finais do agente. |

## **3\. Backlog do Projeto (Epics)**

Estrutura sugerida para importação no Cowork:

### **Epic 1: Configuração do Ambiente e Dados**

* **Task 1.1:** Baixar e extrair ptcg\_engine.zip e arquivos CSV.  
* **Task 1.2:** Criar pipeline de leitura e limpeza do EN Card Data.csv para uso no código Python.  
* **Task 1.3:** Configurar repositório Git com estrutura base (/data, /src/engine\_wrapper, /src/agents).

### **Epic 2: Integração com Simulador Oficial**

* **Task 2.1:** Implementar wrapper em Python para interagir com o motor do Kaggle (receber observações e retornar ações legais).  
* **Task 2.2:** Criar script de validação local rodando partidas *self-play* entre dois agentes com movimentos aleatórios.

### **Epic 3: Desenvolvimento de Agentes**

* **Task 3.1:** Desenvolver *Baseline Agent* baseado em heurísticas (priorizar ataques de maior dano, evoluir sempre que possível).  
* **Task 3.2:** Modelar o espaço de estado (State Space) e espaço de ação (Action Space) para frameworks de RL (ex: Ray RLlib ou Stable Baselines3).  
* **Task 3.3:** Iniciar rotina de treinamento por Reinforcement Learning (PPO).

### **Epic 4: Deck Building e Estratégia**

* **Task 4.1:** Analisar metadados para construir deck.csv inicial otimizado para o agente heurístico.  
* **Task 4.2:** Avaliar viabilidade de treinar o agente para adaptar a estratégia com base em diferentes decks.

## **4\. Instruções Mestras para o Claude / IA de Código (Prompt)**

Use este prompt sempre que iniciar uma sessão de codificação com sua IA assistente para manter o contexto do projeto:  
`[PROMPT MESTRE DE CONTEXTO]`  
`Você atua como Engenheiro Chefe de Machine Learning e Arquiteto de Software. Nosso objetivo é desenvolver um Agente de IA para a competição Kaggle "Pokemon TCG AI Battle".` 

`Restrições e Contexto Crítico:`  
``1. NÃO criaremos um motor de jogo do zero. O Kaggle disponibilizou o motor oficial (`ptcg_engine.zip`). Nosso foco é criar um wrapper/ambiente compatível para treinamento de RL usando esse motor, interagindo com os estados e ações legais fornecidas por ele.``  
``2. A submissão exige um arquivo `submission.tar.gz` (max 197.7 MiB) contendo um `main.py` no diretório raiz e um `deck.csv`.``  
``3. Os dados das cartas estão em formato estruturado (`EN Card Data.csv`), contendo detalhes de HP, ataques, resistências, etc.``  
`4. O ambiente do Kaggle rodará nosso código com limites de recursos (11.8 GiB HDD, 12.2 GiB RAM, 2 vCPUs).`  
`5. O foco atual é seguir as tarefas estruturadas no nosso gerenciador de projetos (Cowork), passando de heurísticas simples para Deep Reinforcement Learning.`

`Sempre garanta que qualquer código gerado considere que o motor apenas apresenta "legal moves" e o agente deve retornar o índice da opção selecionada.`  
    