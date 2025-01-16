-- 1. Agrupamento numérico e sumarização com cálculos de interesse
WITH agrupamento AS (
    SELECT
        Codigo,
        SUM(Qtde_Teorica) AS Total_Qtde_Teorica,
        AVG(Part) AS Media_Part,
        MAX(Data) AS Ultima_Data, -- Mantém a última data para cálculos posteriores
        COUNT(*) AS Frequencia_Transacoes -- Conta o número de transações por código
    FROM
        tabela_ibov
    GROUP BY
        Codigo
),

-- 2. Adicionar métricas baseadas em dados de negócios
metricas_negocio AS (
    SELECT
        Codigo,
        Total_Qtde_Teorica,
        Media_Part,
        Ultima_Data,
        Frequencia_Transacoes,
        DATE_FORMAT(current_date, 'yyyyMMdd') AS Data_Processamento, -- Formata a data de processamento sem hífens
        CASE
            WHEN Media_Part > 1 THEN 'Alta'
            WHEN Media_Part BETWEEN 0.5 AND 1 THEN 'Moderada'
            ELSE 'Baixa'
        END AS Nivel_Participacao -- Classifica o nível de participação média
    FROM
        agrupamento
),

-- 3. Cálculo com campos de data e métricas adicionais
calculo_data AS (
    SELECT
        Codigo,
        Total_Qtde_Teorica,
        Media_Part,
        Frequencia_Transacoes,
        Nivel_Participacao,
        Ultima_Data,
        Data_Processamento,
        DATEDIFF(current_date, TO_DATE(Ultima_Data, 'yyyyMMdd')) AS Dias_Desde_Ultima_Transacao
    FROM
        metricas_negocio
)

-- Resultado final com informações agregadas e a data de processamento
SELECT
    Codigo,
    Total_Qtde_Teorica,
    Media_Part,
    Frequencia_Transacoes,
    Nivel_Participacao,
    Ultima_Data,
    Dias_Desde_Ultima_Transacao,
    Data_Processamento
FROM
    calculo_data;
