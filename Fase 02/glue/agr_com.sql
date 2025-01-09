SELECT
    codigo,
    acao,
    tipo,
    qtde_teorica,
    part,
    CAST(date AS DATE) AS Data,
    COALESCE(part - LAG(part, 1) OVER (PARTITION BY codigo ORDER BY date), 0) AS Part_Change,
    CASE
        WHEN LAG(part, 1) OVER (PARTITION BY codigo ORDER BY date) IS NULL THEN 'Sem comparacao'
        ELSE 'Comparado'
    END AS Status_Comparacao
FROM dados_ibovespa
ORDER BY codigo, date;