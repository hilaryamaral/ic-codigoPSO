import numpy as np
import pandas as pd
import time
import os

rho = 7850  # densidade (kg/m³)
F = 10000  # força aplicada (N)
sigma_adm = 250e6  # tensão admissível (Pa)

def func(x):
    A = x[:, 0]  # área
    L = x[:, 1]  # comprimento
    peso = rho * A * L
    sigma = F / A
    penal = np.maximum(0, sigma - sigma_adm)
    return peso + 1e10 * penal**2

num_particulas = 30
num_iter = 100
w = 0.7  # coeficiente de inércia
c1 = 1.5
c2 = 1.5
lim_inf = np.array([0.0001, 0.5])  # área, comprimento
lim_sup = np.array([0.01, 5.0])

NUM_RODADAS = 30
linhas_historico = []   # rodada, iteracao, melhor_global
linhas_resultados = []  # rodada, massa_final
linhas_tempos = []      # rodada, tempo

for rodada in range(1, NUM_RODADAS + 1):

    # Inicialização das partículas
    posicoes = np.random.uniform(
        lim_inf,
        lim_sup,
        (num_particulas, 2)
    )
    velocidades = np.zeros((num_particulas, 2))
    # Melhor posição individual
    pbest = posicoes.copy()
    pbest_valor = func(pbest)
    # Melhor posição global
    gbest = pbest[np.argmin(pbest_valor)]
    gbest_valor = np.min(pbest_valor)
    # Estruturas para Coleta de Dados
    historico_gbest = []
    historico_media_pso = []
    tempos_iteracoes = []
    tempo_inicio_total = time.perf_counter()
    # Loop principal do PSO
    for _ in range(num_iter):
        t_inicio_iter = time.perf_counter()
        r1 = np.random.rand(num_particulas, 2)
        r2 = np.random.rand(num_particulas, 2)
        velocidades = (
            w * velocidades
            + c1 * r1 * (pbest - posicoes)
            + c2 * r2 * (gbest - posicoes)
        )
        posicoes += velocidades
        # Mantém as partículas dentro dos limites
        posicoes = np.clip(posicoes, lim_inf, lim_sup)
        # Avalia as novas posições
        valores = func(posicoes)
        # Atualiza o melhor individual
        mask = valores < pbest_valor
        pbest[mask] = posicoes[mask]
        pbest_valor[mask] = valores[mask]
        # Atualiza o melhor global
        idx_melhor = np.argmin(pbest_valor)
        gbest = pbest[idx_melhor].copy()
        gbest_valor = pbest_valor[idx_melhor]
        # Coleta de métricas da iteração atual
        historico_gbest.append(gbest_valor)
        historico_media_pso.append(np.mean(valores))
        t_fim_iter = time.perf_counter()
        tempos_iteracoes.append(t_fim_iter - t_inicio_iter)
    tempo_total = time.perf_counter() - tempo_inicio_total

    print(f"[PSO] Rodada {rodada}/{NUM_RODADAS} - massa final: {gbest_valor:.4f} kg "
          f"- tempo: {tempo_total:.4f} s")

    
    for i, melhor in enumerate(historico_gbest, start=1):
        linhas_historico.append({"rodada": rodada, "iteracao": i, "melhor_global": melhor})

    linhas_resultados.append({"rodada": rodada, "massa_final": gbest_valor})
    linhas_tempos.append({"rodada": rodada, "tempo": tempo_total})


pd.DataFrame(linhas_historico).to_csv("historico_iteracoes_pso.csv", index=False)
pd.DataFrame(linhas_resultados).to_csv("resultados_finais_pso.csv", index=False)
pd.DataFrame(linhas_tempos).to_csv("tempos_pso.csv", index=False)

print("\nArquivos salvos: historico_iteracoes_pso.csv, resultados_finais_pso.csv, tempos_pso.csv")
