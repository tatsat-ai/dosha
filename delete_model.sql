-- Delete scores for gemma-4-31b-it responses
DELETE FROM Scores
WHERE response_id IN (
    SELECT response_id FROM Responses
    WHERE model_id = 'gemma-4-31b-it'
);

-- Delete the responses
DELETE FROM Responses WHERE model_id = 'gemma-4-31b-it';

-- Delete G vector and dosha vector
DELETE FROM GVectors      WHERE model_id = 'gemma-4-31b-it';
DELETE FROM DoshaVectors  WHERE model_id = 'gemma-4-31b-it';

-- Delete the model row itself
DELETE FROM Models WHERE model_id = 'gemma-4-31b-it';