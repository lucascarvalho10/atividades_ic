# Adicionando uma linha de cada vez:
new_data = {
    "run": "novo_nome",
    "perda": 0.015,
    "lr": 0.02,
    "alpha": 0.2,
    "numero_pontos": 3000
}

df.loc[len(df)] = new_data
# Salvando como JSON
df.to_json('dados_otimizacao.json', orient='records')

# Function to generate random values for parameters
def generate_random_values():
    lr = 10 ** np.random.uniform(-5, 0)  
    alpha = np.random.uniform(0.1, 0.4)  
    numero_pontos = np.random.randint(1000, 3501)  
    numero_neuronios = np.random.randint(2, 16)  
    numero_camadas = np.random.randint(3, 9)
    return [lr, alpha, numero_pontos, numero_neuronios, numero_camadas]

# Create a dictionary with 5 lists of random parameter values
parameters_dict = {
    f"experiment_{i+1}": generate_random_values() for i in range(100)
}

for key, values in parameters_dict.items():
    print(f"{key}: lr={values[0]}, alpha={values[1]}, numero_pontos={values[2]}, numero_neuronios={values[3]}, numero_camadas={values[4]}")


for key, values in parameters_dict.items():
    # Criando o array completo com os valores intermediários
    rede = [3] + [values[3]] * values[4] + [2]
    pinn_ns = medicao_qualidade(1, 1, 1, 1000, values[2], rede, 300, 400, values[1], 700, values[0])
    
    new_data = {
        "run": key,
        "perda": pinn_ns.perda_historico[-1],
        "lr": pinn_ns.learning_rate,
        "alpha": pinn_ns.alpha,
        "numero_pontos": pinn_ns.pontos_no_dominio,
        "numero_neuronios": values[3],
        "numero_camadas": values[4]
    }

    df.loc[len(df)] = new_data
    # Salvando como JSON
    df.to_json('dados_otimizacao.json', orient='records')

new_data = {
    "run": "lalala",
    "perda": pinn_ns.perda_historico[-1],
    "lr": pinn_ns.learning_rate,
    "alpha": pinn_ns.alpha,
    "numero_pontos": pinn_ns.pontos_no_dominio
}

df.loc[len(df)] = new_data
# Salvando como JSON
df.to_json('dados_otimizacao.json', orient='records')