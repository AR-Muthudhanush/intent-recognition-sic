import React, { useState } from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Card, Title, Paragraph, Button, Snackbar } from 'react-native-paper';

const JSONViewerScreen = ({ route }: any) => {
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const data = route?.params?.data || {
    intent: 'click',
    confidence: 0.95,
    target: {
      type: 'button',
      attribute: null,
      label: 'submit',
      index: null,
      relation: null,
      reference: null,
    },
  };

  const jsonString = JSON.stringify(data, null, 2);

  const handleCopy = async () => {
    try {
      // In a real app, would use react-native-clipboard
      setSnackbarMessage('JSON copied to clipboard');
      setShowSnackbar(true);
    } catch (error) {
      setSnackbarMessage('Failed to copy');
      setShowSnackbar(true);
    }
  };

  const handleDownload = async () => {
    try {
      // In a real app, would download the JSON file
      setSnackbarMessage('JSON download started');
      setShowSnackbar(true);
    } catch (error) {
      setSnackbarMessage('Failed to download');
      setShowSnackbar(true);
    }
  };

  const handleShare = async () => {
    try {
      // In a real app, would use react-native-share
      setSnackbarMessage('Share dialog opened');
      setShowSnackbar(true);
    } catch (error) {
      setSnackbarMessage('Failed to share');
      setShowSnackbar(true);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>JSON Output</Title>
          <Paragraph style={styles.jsonText}>{jsonString}</Paragraph>
        </Card.Content>
      </Card>

      <View style={styles.buttonContainer}>
        <Button
          mode="contained"
          onPress={handleCopy}
          style={styles.button}
        >
          Copy
        </Button>
        <Button
          mode="contained"
          onPress={handleDownload}
          style={styles.button}
        >
          Download
        </Button>
        <Button
          mode="contained"
          onPress={handleShare}
          style={styles.button}
        >
          Share
        </Button>
      </View>

      <Card style={styles.detailCard}>
        <Card.Content>
          <Title>Field Descriptions</Title>

          <Paragraph style={styles.fieldLabel}>intent</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            The recognized UI action (e.g., click, scroll, drag)
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>confidence</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            Confidence score (0.0 - 1.0) for the prediction
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>target.type</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            The UI element type (e.g., button, input, checkbox)
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>target.attribute</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            Visual attributes (e.g., color, size, state)
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>target.label</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            Text label on the UI element
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>target.index</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            Position index (e.g., first, second, last)
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>target.relation</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            Spatial relationship (e.g., above, below, beside)
          </Paragraph>

          <Paragraph style={styles.fieldLabel}>target.reference</Paragraph>
          <Paragraph style={styles.fieldDescription}>
            Reference element for relative positioning
          </Paragraph>
        </Card.Content>
      </Card>

      <Snackbar
        visible={showSnackbar}
        onDismiss={() => setShowSnackbar(false)}
        duration={3000}
      >
        {snackbarMessage}
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
  jsonText: {
    fontFamily: 'monospace',
    fontSize: 12,
    marginTop: 8,
    backgroundColor: '#f0f0f0',
    padding: 12,
    borderRadius: 4,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  button: {
    flex: 1,
    marginHorizontal: 4,
  },
  detailCard: {
    marginBottom: 16,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    marginTop: 12,
    color: '#6200ee',
  },
  fieldDescription: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
});

export default JSONViewerScreen;
