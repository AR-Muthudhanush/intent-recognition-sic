import React, { useState } from 'react';
import { View, ScrollView, StyleSheet, ActivityIndicator } from 'react-native';
import { TextInput, Button, Card, Title, Paragraph, Snackbar } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import InferenceService from '../services/InferenceService';

const AnalyzeScreen = ({ navigation }: any) => {
  const [command, setCommand] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showSnackbar, setShowSnackbar] = useState(false);

  const handleAnalyze = async () => {
    if (!command.trim()) {
      setError('Please enter a command');
      setShowSnackbar(true);
      return;
    }

    setLoading(true);
    try {
      const prediction = await InferenceService.predict(command);
      setResult(prediction);

      const history = await AsyncStorage.getItem('analyzeHistory');
      const historyArray = history ? JSON.parse(history) : [];

      historyArray.unshift({
        id: Date.now().toString(),
        command,
        prediction,
        timestamp: new Date().toISOString(),
      });

      if (historyArray.length > 100) {
        historyArray.pop();
      }

      await AsyncStorage.setItem('analyzeHistory', JSON.stringify(historyArray));
    } catch (err: any) {
      setError(err.message || 'Failed to analyze command');
      setShowSnackbar(true);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!result) return;

    try {
      // In a real app, would use react-native-clipboard
      setError('JSON copied to clipboard');
      setShowSnackbar(true);
    } catch (err) {
      setError('Failed to copy');
      setShowSnackbar(true);
    }
  };

  const handleShare = async () => {
    if (!result) return;

    try {
      const text = JSON.stringify(result, null, 2);
      // In a real app, would use react-native-share
      setError('Share functionality');
      setShowSnackbar(true);
    } catch (err) {
      setError('Failed to share');
      setShowSnackbar(true);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>Enter Command</Title>
          <TextInput
            label="UI Command"
            value={command}
            onChangeText={setCommand}
            placeholder="e.g., 'click the submit button' or '제출 버튼을 클릭해'"
            multiline
            numberOfLines={4}
            style={styles.input}
          />
        </Card.Content>
      </Card>

      <Button
        mode="contained"
        onPress={handleAnalyze}
        loading={loading}
        disabled={loading}
        style={styles.analyzeButton}
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </Button>

      {result && (
        <Card style={styles.resultCard}>
          <Card.Content>
            <Title>Prediction Result</Title>

            <Paragraph style={styles.resultLabel}>Intent</Paragraph>
            <Paragraph style={styles.resultValue}>{result.intent}</Paragraph>

            <Paragraph style={styles.resultLabel}>Confidence</Paragraph>
            <Paragraph style={styles.resultValue}>{(result.confidence * 100).toFixed(2)}%</Paragraph>

            <Paragraph style={styles.resultLabel}>Target Information</Paragraph>

            {result.target.type && (
              <>
                <Paragraph style={styles.targetLabel}>Type</Paragraph>
                <Paragraph style={styles.targetValue}>{result.target.type}</Paragraph>
              </>
            )}

            {result.target.attribute && (
              <>
                <Paragraph style={styles.targetLabel}>Attribute</Paragraph>
                <Paragraph style={styles.targetValue}>{result.target.attribute}</Paragraph>
              </>
            )}

            {result.target.relation && (
              <>
                <Paragraph style={styles.targetLabel}>Relation</Paragraph>
                <Paragraph style={styles.targetValue}>{result.target.relation}</Paragraph>
              </>
            )}

            {result.target.reference && (
              <>
                <Paragraph style={styles.targetLabel}>Reference</Paragraph>
                <Paragraph style={styles.targetValue}>{result.target.reference}</Paragraph>
              </>
            )}
          </Card.Content>

          <Card.Actions>
            <Button onPress={handleCopy}>Copy JSON</Button>
            <Button onPress={handleShare}>Share</Button>
          </Card.Actions>
        </Card>
      )}

      <Snackbar
        visible={showSnackbar}
        onDismiss={() => setShowSnackbar(false)}
        duration={3000}
      >
        {error}
      </Snackbar>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  card: {
    marginBottom: 16,
  },
  input: {
    marginTop: 8,
  },
  analyzeButton: {
    marginBottom: 16,
  },
  resultCard: {
    marginBottom: 16,
  },
  resultLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
  },
  resultValue: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  targetLabel: {
    fontSize: 12,
    color: '#999',
    marginLeft: 8,
  },
  targetValue: {
    fontSize: 14,
    marginLeft: 16,
    marginBottom: 4,
  },
});

export default AnalyzeScreen;
