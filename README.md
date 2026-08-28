import numpy as np
import time

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

# Cálculo estatístico dos tempos
media_tempo_iter = np.mean(tempos_iteracoes)
desvio_tempo_iter = np.std(tempos_iteracoes)

# Exibição dos Dados Coletados (Iteração x Valores x Tempo)
print("Iteração | Melhor Global (kg) | Média Enxame (kg)  | Tempo (s)")
print("-" * 65)
for i in range(num_iter):
    print(f"{i+1:8d} | {historico_gbest[i]:18.4f} | {historico_media_pso[i]:18.4f} | {tempos_iteracoes[i]:.6f}")

print("=" * 65)
print("RESUMO FINAL (PSO):")
print(f"Melhor Área (A): {gbest[0]:.6f} m²")
print(f"Melhor Comprimento (L): {gbest[1]:.4f} m")
print(f"Massa mínima (Custo): {func(gbest.reshape(1, -1))[0]:.4f} kg")
print("-" * 65)
print(f"Tempo Total: {tempo_total:.4f} s")
print(f"Média de tempo por iteração: {media_tempo_iter:.6f} s ({media_tempo_iter*1000:.3f} ms)")
print(f"Desvio Padrão do tempo: {desvio_tempo_iter:.6f} s ({desvio_tempo_iter*1000:.3f} ms)")
print("=" * 65)
