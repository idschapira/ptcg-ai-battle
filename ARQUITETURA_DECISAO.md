# Arquitetura de Decisão — PTCG AI Battle

O "agente de IA por trás das decisões" **no runtime da competição** é uma **rede neural (policy + value) com busca guiada**, não um LLM. LLM e MCP são ferramentas de *tempo de desenvolvimento* (copiloto de engenharia + destilador de conhecimento offline). Este documento define a arquitetura do cérebro do agente e como ele reaproveita 100% do que já foi construído nas Sprints 1–2.

---

## 1. Restrição que define tudo

O `agent(obs_dict) -> list[int]` roda no sandbox do Kaggle: **sem rede, 2 vCPU, orçamento de milissegundos por jogada, pacote ≤ 197.7 MiB**. Consequências diretas:

- Nada de chamar API de LLM ou MCP em runtime — o cérebro é autocontido.
- Modelo precisa ser pequeno e rápido (uma rede densa/pequena cabe com folga em 2 MiB; o gargalo é *tempo*, não tamanho).
- Qualquer "inteligência de texto" (efeitos de carta) tem de estar **pré-computada** e embutida como tabela.

## 2. Reality check: PTCG é jogo de informação imperfeita e estocástico

Diferente de xadrez/Go, aqui há:

- **Informação oculta:** mão do oponente, ordem dos dois decks, prizes virados.
- **Aleatoriedade:** cara-ou-coroa, embaralhamentos, compra.

Isso significa que **MCTS "puro" estilo AlphaZero não se aplica direto** (ele pressupõe informação perfeita e determinismo). Caminhos corretos:

- **Determinized / Information-Set MCTS (ISMCTS):** amostrar uma "determinização" plausível do estado oculto por rollout.
- **A verificar no código:** como o `search_step` do engine preenche o estado oculto (ele amostra uma determinização? usa a mão real?). Isso decide quanto de ISMCTS precisamos implementar por cima.
- Dado o orçamento de 2 vCPU/ms, a aposta pragmática é **política forte + busca rasa**, não busca profunda.

## 3. Arquitetura em 3 camadas (incremental, cada uma já joga)

**Camada 0 — Agente heurístico (Sprint 3).**
Regras + *features de efeito destiladas*. Já supera o random, é o baseline do Gate B, e vira a função de avaliação/rollout inicial das camadas seguintes.

**Camada 1 — Rede Policy + Value (Sprints 4–5).**
- *Entrada:* estado codificado (ver §4) + máscara de ações legais.
- *Saída:* política sobre os *slots* de `obs.select.option` + valor (prob. de vitória).
- *Treino:* primeiro por imitação do heurístico/self-play (arranque estável), depois RL.

**Camada 2 — Busca guiada (Sprints 5–6).**
ISMCTS/rollouts rasos via `search_begin`, com *prior* vindo da policy net e avaliação de folha vinda da value net. **Fallback:** se o orçamento de tempo apertar, cai para `argmax` da policy net (sem busca). Segurança de runtime garantida.

## 4. Encoding de estado (consome o Epic 1 direto)

Vetor de features montado a partir de `Observation` + `CardIndex`:

- **Seu campo/oponente:** ativo e banco (id, hp/maxHp, energias por tipo — já unificadas ao namespace do engine `COLORLESS=0…TEAM_ROCKET=11`), condições especiais (poison/burn/asleep/paralyzed/confused).
- **Contagens:** mão (própria = ids; oponente = contagem), deck restante, prizes restantes, descarte.
- **Contexto de turno:** turn, supporterPlayed, energyAttached, retreated, stadium.
- **Features de efeito destiladas** (ver §6) das cartas relevantes em jogo/mão.

A unificação de energia que o Epic 1 já fez permite comparar `bridge_attack_energy` direto com `Pokemon.energies` — o encoder ganha "posso pagar este ataque?" quase de graça.

## 5. Espaço de ação (pointer-style, não vocabulário fixo)

A ação é um **índice em `obs.select.option`**, mas a lista é de tamanho variável e as options têm semânticas diferentes (PLAY/ATTACK/CARD/ENERGY/…). Portanto a *policy head* pontua **cada option enriquecida** (pointer network), com máscara — e não um vetor de ações de tamanho fixo. O `EnvironmentWrapper.enrich()` já produz features por option (resolve PLAY/ATTACK/CARD com fallback None): é exatamente o insumo dessa head.

## 6. Destilação de efeitos (LLM offline, uma vez, custo zero em runtime)

Alavancagem alta e baixo risco.

- **Entrada:** os ~1.811 textos de efeito (`dim_attack.effect`) + 255 skills (`dim_skill`).
- **Processo:** Claude, offline no dev, converte texto livre → schema estruturado, ex.:
  `{effect_type ∈ [damage_bonus, heal, energy_accel, energy_discard, draw, search, status_inflict(poison/burn/sleep/paralyze/confuse), switch, disrupt, prize_manip, conditional_flip, self_damage, ...], magnitude, target, condition, coin_flip?}`.
- **Saída:** nova tabela `dim_effect` (Parquet, keyed por attack_id/skill) — leve, embutida na submissão.
- **Consumidores:** função heurística (Camada 0) **e** vetor de features da NN (Camada 1).
- **Verificação (LLM alucina):** cruzar `magnitude` com `dim_attack.damage_base` onde aplicável; auditoria amostral manual de ~30 cartas; marcar itens de baixa confiança para revisão.

## 7. Loop de treino (reaproveita o harness de self-play)

`selfplay.py` já roda partidas e conta resultados. Vira o gerador de dados do AlphaZero-like:

1. Self-play com o agente atual (Camada 1/2) registrando `(estado, política_da_busca π, resultado z)`.
2. Treinar policy+value nesses triplets.
3. Avaliar novo agente vs. melhor anterior na arena interna; promover se ganhar com margem.
4. Repetir. Gate C = política ≥ melhor baseline.

## 8. Impacto no que já foi construído: **aditivo, zero retrabalho**

| Artefato existente | Papel na nova arquitetura |
| :-- | :-- |
| `CardIndex` + `dim_card`/`dim_attack`/`dim_skill` | Base do encoder de estado e da avaliação. Reuso direto. |
| Unificação de energia (namespace do engine) | Comparação direta custo ↔ `Pokemon.energies` no encoder. Bônus. |
| `EnvironmentWrapper` (`legal_option_indices`, `enrich`) | Enumeração de ações da busca + featurização por option (pointer head). **Peça central.** |
| `selfplay.py` | Vira o gerador de dados do self-play de RL. Reuso + extensão. |
| `RandomAgent` | Permanece como oponente/baseline e sanity check. |
| `build_submission.py` | Inalterado; pesos da NN são minúsculos vs. 197.7 MiB. |
| Novo: `dim_effect` (destilação) | Enriquece heurística **e** features da NN. |

Ou seja: as Sprints 1–2 são exatamente a fundação que essa arquitetura exige. O que muda é o *conteúdo* do Epic 4 (Task 4.1 passa a incluir features de efeito + pointer head; Task 4.3 vira self-play AlphaZero-like, com PPO como fallback), e a Sprint 3 ganha a destilação de efeitos.

## 9. Decisão em aberto (para o Ilan)

Dois "cérebros" viáveis dentro do orçamento de 2 vCPU/ms:

- **(a) Policy-net-first** — rápido, simples, entrega cedo; busca rasa opcional depois. *Recomendado começar por aqui.*
- **(b) ISMCTS pesado** — teto de força maior, porém arriscado no orçamento de tempo e na informação oculta.

Sugestão: começar em **(a)** e só adicionar busca (Camada 2) se o profiling mostrar folga de tempo por jogada.

---

*Nota sobre o `RandomAgent`: a variação `k ∈ [minCount, maxCount]` (em vez do `maxCount` fixo do sample) é desejável para o self-play — exercita mais caminhos do engine. Para benchmarks reprodutíveis, dá para expor um modo `maxCount` fixo via flag.*
