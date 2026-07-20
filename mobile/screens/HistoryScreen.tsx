import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, FlatList } from 'react-native';
import { Card, Title, Paragraph, Button, IconButton, Searchbar } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import moment from 'moment';

const HistoryScreen = ({ navigation }: any) => {
  const [history, setHistory] = useState<any[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadHistory();
    const unsubscribe = navigation.addListener('focus', () => {
      loadHistory();
    });
    return unsubscribe;
  }, [navigation]);

  const loadHistory = async () => {
    try {
      const savedHistory = await AsyncStorage.getItem('analyzeHistory');
      const data = savedHistory ? JSON.parse(savedHistory) : [];
      setHistory(data);
      setFilteredHistory(data);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (!query.trim()) {
      setFilteredHistory(history);
    } else {
      const filtered = history.filter(item =>
        item.command.toLowerCase().includes(query.toLowerCase())
      );
      setFilteredHistory(filtered);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const updated = history.filter(item => item.id !== id);
      setHistory(updated);
      setFilteredHistory(updated);
      await AsyncStorage.setItem('analyzeHistory', JSON.stringify(updated));
    } catch (error) {
      console.error('Failed to delete history item:', error);
    }
  };

  const handleClearAll = async () => {
    try {
      await AsyncStorage.setItem('analyzeHistory', JSON.stringify([]));
      setHistory([]);
      setFilteredHistory([]);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const renderItem = ({ item }: any) => (
    <Card style={styles.historyCard}>
      <Card.Content>
        <Title style={styles.command}>{item.command}</Title>
        <Paragraph style={styles.timestamp}>
          {moment(item.timestamp).format('YYYY-MM-DD HH:mm:ss')}
        </Paragraph>
        <Paragraph>
          Intent: <Paragraph style={styles.intent}>{item.prediction.intent}</Paragraph>
          Confidence: <Paragraph style={styles.confidence}>
            {(item.prediction.confidence * 100).toFixed(2)}%
          </Paragraph>
        </Paragraph>
      </Card.Content>
      <Card.Actions>
        <Button onPress={() => navigation.navigate('JSONViewer', { data: item.prediction })}>
          View JSON
        </Button>
        <IconButton
          icon="delete"
          onPress={() => handleDelete(item.id)}
        />
      </Card.Actions>
    </Card>
  );

  return (
    <View style={styles.container}>
      <Searchbar
        placeholder="Search history"
        onChangeText={handleSearch}
        value={searchQuery}
        style={styles.searchbar}
      />

      {history.length > 0 && (
        <Button
          mode="text"
          onPress={handleClearAll}
          color="red"
          style={styles.clearButton}
        >
          Clear All History
        </Button>
      )}

      {filteredHistory.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Paragraph style={styles.emptyText}>
            {history.length === 0 ? 'No history yet' : 'No matches found'}
          </Paragraph>
        </View>
      ) : (
        <FlatList
          data={filteredHistory}
          renderItem={renderItem}
          keyExtractor={item => item.id}
          style={styles.list}
          scrollEnabled={false}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  searchbar: {
    marginBottom: 16,
  },
  clearButton: {
    marginBottom: 16,
  },
  list: {
    flex: 1,
  },
  historyCard: {
    marginBottom: 12,
  },
  command: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  timestamp: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  intent: {
    fontWeight: 'bold',
    color: '#6200ee',
  },
  confidence: {
    fontWeight: 'bold',
    color: '#4caf50',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
    color: '#999',
  },
});

export default HistoryScreen;
