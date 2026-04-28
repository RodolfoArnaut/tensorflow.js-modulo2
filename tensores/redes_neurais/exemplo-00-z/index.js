import tf from '@tensorflow/tfjs';

// ==========================================
// CASO REAL: CLASSIFICAÇÃO DE NÍVEL DE CLIENTE (SAAS)
// Objetivo: Classificar clientes em Basic, Medium ou Premium
// Parâmetros: Qtd Usuários, Qtd Canais, IA Generativa (Sim/Não)
// ==========================================

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Entrada: 3 posições [Qtd Usuários, Qtd Canais, IA Generativa]
    model.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
    
    // Camada Oculta para processar os pesos cruzados
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));

    // Saída: 3 categorias [Basic, Medium, Premium]
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    console.log("Treinando o modelo de análise de clientes...");
    await model.fit(inputXs, outputYs, {
        epochs: 500,
        verbose: 0
    });

    return model;
}

// --- PREPARAÇÃO DOS DADOS (DATASET DE EXEMPLO) ---
// Formato: [Usuários (0 a 1), Canais (0 a 1), IA Gen (0 ou 1)]
// Obs: Usuários peso menor, Canais peso maior.

const dadosTreino = [
    [0.1, 0.1, 0], // Poucos usuários, Poucos canais, Sem IA -> Basic
    [0.4, 0.2, 0], // Médio usuários, Poucos canais, Sem IA -> Basic
    [0.2, 0.5, 0], // Poucos usuários, Médio canais, Sem IA -> Medium
    [0.8, 0.4, 0], // Muitos usuários, Médio canais, Sem IA -> Medium
    [0.3, 0.8, 0], // Poucos usuários, Muitos canais, Sem IA -> Premium (Canais pesam mais!)
    [0.1, 0.4, 1], // Poucos usuários, Médio canais, Com IA -> Premium (IA impulsiona!)
    [0.9, 0.9, 1], // Topo de linha -> Premium
    [0.1, 0.2, 0], // Outro Basic
];

const labelsTreino = [
    [1, 0, 0], // Basic
    [1, 0, 0], // Basic
    [0, 1, 0], // Medium
    [0, 1, 0], // Medium
    [0, 0, 1], // Premium
    [0, 0, 1], // Premium
    [0, 0, 1], // Premium
    [1, 0, 0], // Basic
];

const inputXs = tf.tensor2d(dadosTreino);
const outputYs = tf.tensor2d(labelsTreino);

// --- EXECUÇÃO ---
(async () => {
    const model = await trainModel(inputXs, outputYs);
    const categorias = ["Basic", "Medium", "Premium"];

    // FUNÇÃO DE NORMALIZAÇÃO PARA TESTE
    // Simula: Max 100 usuários, Max 20 canais
    const analisarCliente = (usuarios, canais, temIA) => {
        const u = Math.min(usuarios / 100, 1);
        const c = Math.min(canais / 20, 1);
        const ia = temIA ? 1 : 0;
        return [u, c, ia];
    };

    // TESTE 1: Cliente Pequeno
    const cliente1 = analisarCliente(10, 2, false);
    
    // TESTE 2: Cliente com muitos canais mas poucos usuários
    const cliente2 = analisarCliente(5, 18, false);

    // TESTE 3: Cliente médio que ativou a IA Generativa
    const cliente3 = analisarCliente(30, 8, true);

    const testar = async (dados, label) => {
        const pred = model.predict(tf.tensor2d([dados]));
        const result = await pred.array();
        console.log(`\nAnálise: ${label}`);
        result[0].forEach((prob, i) => {
            console.log(`${categorias[i]}: ${(prob * 100).toFixed(2)}%`);
        });
    };

    await testar(cliente1, "Cliente Pequeno (10 users, 2 canais, Sem IA)");
    await testar(cliente2, "Power User de Canais (5 users, 18 canais, Sem IA)");
    await testar(cliente3, "Upgrade para IA (30 users, 8 canais, Com IA)");
})();