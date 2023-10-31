def medicao_qualidade(dimensao_x, dimensao_y, tempo, pontos_contorno, pontos_dominio, camadas, alpha, epocas):

    def gerar_pontos_contorno(pontos_no_contorno,comprimento_x,comprimento_y,tempo_final):
        pontos_por_lado = pontos_no_contorno//6

        # Lado 1 (x = qualquer, y= 0, t = qualquer)
        x_lado1 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=comprimento_x)
        y_lado1 = 0 * np.ones((pontos_por_lado,1))
        t_lado1 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=tempo_final)

        u_lado1 = 0 * np.ones((pontos_por_lado,1))
        v_lado1 = 0 * np.ones((pontos_por_lado,1))

        # Lado 2 (x = 0, y= qualquer, t = qualquer)
        x_lado2 = 0 * np.ones((pontos_por_lado,1))
        y_lado2 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=comprimento_y)
        t_lado2 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=tempo_final)

        u_lado2 = 0 * np.ones((pontos_por_lado,1))
        v_lado2 = 0 * np.ones((pontos_por_lado,1))

        # Lado 3 (x = 1, y = qualquer, t = qualquer)
        x_lado3 = 1 * np.ones((pontos_por_lado,1))
        y_lado3 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=comprimento_y)
        t_lado3 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=tempo_final)

        u_lado3 = 0 * np.ones((pontos_por_lado,1))
        v_lado3 = 0 * np.ones((pontos_por_lado,1))

        # Lado 4 (x = qualquer, y = 1, t = qualquer)
        x_lado4 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=comprimento_x)
        y_lado4 = 1 * np.ones((pontos_por_lado,1))
        t_lado4 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=tempo_final)

        u_lado4 = 0 * np.ones((pontos_por_lado,1))
        v_lado4 = 0 * np.ones((pontos_por_lado,1))

        # Condicao inicial (x = qualquer, y=qualquer, t = 0)
        x_inicial = np.random.uniform(size=(2*pontos_por_lado,1), low=0, high=comprimento_x)
        y_inicial = np.random.uniform(size=(2*pontos_por_lado,1), low=0, high=comprimento_y)
        t_inicial = 0 * np.ones((2*pontos_por_lado,1))

        # Modificação das condições iniciais para garantir u=0 nos contornos
        u_inicial = 0 * np.ones((2*pontos_por_lado,1))
        v_inicial = np.sin(np.pi * x_inicial / comprimento_x) * np.cos(np.pi * x_inicial / comprimento_y)


        # Juntar todos os lados
        x_todos = np.vstack((x_lado1,x_lado2,x_lado3,x_lado4,x_inicial))
        y_todos = np.vstack((y_lado1,y_lado2,y_lado3,y_lado4,y_inicial))
        t_todos = np.vstack((t_lado1,t_lado2,t_lado3,t_lado4,t_inicial))
        u_todos = np.vstack((u_lado1,u_lado2,u_lado3,u_lado4,u_inicial))
        v_todos = np.vstack((v_lado1,v_lado2,v_lado3,v_lado4,v_inicial))


        # Criar arrays X e Y
        X_contorno = np.hstack((x_todos,y_todos,t_todos))
        Y_contorno = np.hstack((u_todos,v_todos))

        return X_contorno, Y_contorno
        # X contorno - reúne os pontos
        # Y contorno - reúne velocidades
    
    def gerar_pontos_equacao(pontos_no_dominio,comprimento_x,comprimento_y,tempo_final):
        x_dominio = np.random.uniform(size=(pontos_no_dominio,1),low=0,high=comprimento_x)
        y_dominio = np.random.uniform(size=(pontos_no_dominio,1),low=0,high=comprimento_y)
        t_dominio = np.random.uniform(size=(pontos_no_dominio,1),low=0,high=tempo_final)

        X_equacao = np.hstack((x_dominio,y_dominio,t_dominio))

        return X_equacao
    
    comprimento_x = dimensao_x
    comprimento_y = dimensao_y
    tempo_final = tempo

    pontos_no_contorno = pontos_contorno
    pontos_no_dominio = pontos_dominio

    X_contorno, Y_contorno = gerar_pontos_contorno(pontos_no_contorno,comprimento_x,comprimento_y,tempo_final)
    X_equacao = gerar_pontos_equacao(pontos_no_dominio,comprimento_x,comprimento_y,tempo_final)

    def criar_rede_neural(numero_de_neuronios):

        # Criar uma lista de todas as camadas
        camadas = []

        # Para cada camada, adicionar as conexões e a função de ativação
        for i in range(len(numero_de_neuronios)-1):
            camadas.append(nn.Linear(numero_de_neuronios[i],numero_de_neuronios[i+1]))
            camadas.append(nn.Tanh())

        # Remover a última camada, pois é a função de ativação
        camadas.pop()
        #camadas.pop()

        # Criar rede
        return nn.Sequential(*camadas)
    
    numero_de_neuronios = camadas

    rna = criar_rede_neural(numero_de_neuronios)

    def calc_perda_contorno(rna,X_contorno,Y_contorno):
        Y_predito = rna(X_contorno)
        return nn.functional.mse_loss(Y_predito, Y_contorno)

    def calc_residuo(rna,X_equacao):
        x = X_equacao[:,0].reshape(-1, 1)
        y = X_equacao[:,1].reshape(-1, 1)
        t = X_equacao[:,2].reshape(-1, 1)

        # Dois valores preditos pela rede - velocidade u(x,y,t) e velocidade v(x,y,t)
        V = rna(torch.hstack((x, y, t)))
        u = V[:,0].reshape(-1,1)
        v = V[:,1].reshape(-1,1)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]

        residual_u = (u_t + u * u_x + v * u_y - 0.01/np.pi * (u_xx + u_yy))
        residual_v = (v_t + u * v_x + v * v_y - 0.01/np.pi * (u_xx + u_yy))

        return torch.cat((residual_u, residual_v), dim=1)

    def calc_perda_equacao(rna, X_equacao):
        R = calc_residuo(rna, X_equacao)
        residuo = torch.mean(torch.square(R))

        return residuo

    def calc_perda(rna,X_contorno,Y_contorno,X_equacao,alpha=0.2):
        perda_contorno = calc_perda_contorno(rna,X_contorno,Y_contorno)
        perda_equacao = calc_perda_equacao(rna,X_equacao)

        perda = (1-alpha)*perda_contorno + alpha*perda_equacao

        return perda, perda_contorno, perda_equacao

    otimizador = torch.optim.Adam(rna.parameters(),lr=0.01)
    agendador = torch.optim.lr_scheduler.StepLR(otimizador, step_size=1000, gamma=0.9)
    alpha = alpha

    X_equacao = torch.tensor(X_equacao,requires_grad=True,dtype=torch.float)
    X_contorno = torch.tensor(X_contorno,dtype=torch.float)
    Y_contorno = torch.tensor(Y_contorno,dtype=torch.float)

    device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
    X_equacao = X_equacao.to(device)
    X_contorno = X_contorno.to(device)
    Y_contorno = Y_contorno.to(device)
    rna = rna.to(device)

    def calcular_grid(rna, comprimento_x, comprimento_y, tempo_final, nx=101, ny=101,nt=101):
        # Definir grid
        x = np.linspace(0.,comprimento_x,nx)
        y = np.linspace(0.,comprimento_y,ny)
        t = np.linspace(0.,tempo_final,nt)
        [t_grid, y_grid, x_grid] = np.meshgrid(t,y,x)
        x = torch.tensor(x_grid.flatten()[:,None],requires_grad=True,dtype=torch.float).to(device)
        y = torch.tensor(y_grid.flatten()[:,None],requires_grad=True,dtype=torch.float).to(device)
        t = torch.tensor(t_grid.flatten()[:,None],requires_grad=True,dtype=torch.float).to(device)

        # Avaliar modelor
        rna.eval()
        Y_pred = rna(torch.hstack((x,y,t)))

        # Formatar resultados em array
        u_pred = Y_pred.cpu().detach().numpy()[:,0].reshape(x_grid.shape)
        v_pred = Y_pred.cpu().detach().numpy()[:,1].reshape(x_grid.shape)

        return x_grid, y_grid, t_grid, u_pred, v_pred
    
    numero_de_epocas = epocas
    perda_historico = np.zeros(numero_de_epocas)
    perda_contorno_historico = np.zeros(numero_de_epocas)
    perda_equacao_historico = np.zeros(numero_de_epocas)
    epocas = np.array(range(numero_de_epocas))

    # Colocar rede em modo de treinamento
    rna.train()

    # FAZER ITERAÇÃO
    for epoca in epocas:

        # Resortear pontos
        #X_equacao = gerar_pontos_equacao(pontos_no_dominio,comprimento_x,tempo_final)
        #X_equacao = torch.tensor(X_equacao,requires_grad=True,dtype=torch.float).to(device)

        # Inicializar gradientes
        otimizador.zero_grad()

        # Calcular perdas
        perda, perda_contorno, perda_equacao = calc_perda(rna,X_contorno,Y_contorno,X_equacao,alpha=alpha)

        # Backpropagation
        perda.backward()

        # Passo do otimizador
        otimizador.step()
        agendador.step()

        # Guardar logs
        perda_historico[epoca] = perda.item()
        perda_contorno_historico[epoca] = perda_contorno.item()
        perda_equacao_historico[epoca] = perda_equacao.item()

        if epoca%500==0:
            print(f'Epoca: {epoca}, Perda: {perda.item()} (Contorno: {perda_contorno.item()}, Equacao: {perda_equacao.item()})')

    x_grid, y_grid, t_grid, u_pred, v_pred = calcular_grid(rna, comprimento_x, comprimento_y, tempo_final)

    return perda_historico[perda]