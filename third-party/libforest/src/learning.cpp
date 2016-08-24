#include "libforest/learning.h"
#include "libforest/data.h"
#include "libforest/classifiers.h"
#include "libforest/util.h"
#include "fastlog.h"
#include "mcmc.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

using namespace libf;

#define ENTROPY(p) (-(p)*fastlog2(p))
#define SIGMOID(x) (1.0f/(1.0f + std::exp(x)))

static std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
/// DecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

/**
 * A histogram over the class labels. We use this for training
 */
class EfficientEntropyHistogram {
private:
    /**
     * The number of classes in this histogram
     */
    unsigned char bins;

    /**
     * The actual histogram
     */
    int* histogram;
    float* weighted_histogram;

    /**
     * The integral over the entire histogram
     */
    float mass;
    float weighted_mass;

    /**
     * The entropies for the single bins
     */
    float* entropies;

    /**
     * The total entropy
     */
    float totalEntropy;

    float* class_prior;

public:
    /**
     * Default constructor
     */
    EfficientEntropyHistogram() : bins(0), histogram(0), weighted_histogram(0), mass(0), weighted_mass(0), entropies(0), totalEntropy(0), class_prior(0) { }
    EfficientEntropyHistogram(int _classCount) : bins(_classCount), histogram(0), weighted_histogram(0), mass(0), weighted_mass(0), entropies(0), totalEntropy(0), class_prior(0) { resize(_classCount); }

    /**
     * Copy constructor
     */
    EfficientEntropyHistogram(const EfficientEntropyHistogram & other)
    {
        resize (other.bins);
        for (int i = 0; i < bins; i++)
        {
            set(i, other.at(i));
            weighted_histogram[i] = other.weighted_histogram[i];
            class_prior[i] = other.class_prior[i];
        }
        mass = other.mass;
        weighted_mass = other.weighted_mass;
    }

    /**
     * Assignment operator
     */
    EfficientEntropyHistogram & operator= (const EfficientEntropyHistogram &other)
    {
        // Prevent self assignment
        if (this != &other)
        {
            if (other.bins != bins)
            {
                resize (other.bins);
            }
            for (int i = 0; i < bins; i++)
            {
                set(i, other.at(i));
                weighted_histogram[i] = other.weighted_histogram[i];
                entropies[i] = other.entropies[i];
                class_prior[i] = other.class_prior[i];
            }
            mass = other.mass;
            weighted_mass = other.weighted_mass;
            totalEntropy = other.totalEntropy;
        }
        return *this;
    }

    /**
     * Destructor
     */
    ~EfficientEntropyHistogram()
    {
        if (histogram != 0)
        {
            delete[] histogram;
        }
        if (weighted_histogram != 0)
        {
            delete[] weighted_histogram;
        }
        if (entropies != 0)
        {
            delete[] entropies;
        }
        if (class_prior != 0)
        {
            delete[] class_prior;
        }
    }

    /**
     * Resizes the histogram to a certain size
     */
    void resize(int _classCount)
    {
        // Release the current histogram
        if (histogram != 0)
        {
            delete[] histogram;
            histogram = 0;
        }
        if (weighted_histogram != 0)
        {
          delete[] weighted_histogram;
          weighted_histogram = 0;
        }
        if (entropies != 0)
        {
            delete[] entropies;
            entropies = 0;
        }
        if (class_prior != 0)
        {
          delete[] class_prior;
          class_prior = 0;
        }

        // Only allocate a new histogram, if there is more than one class
        if (_classCount > 0)
        {
            histogram = new int[_classCount];
            weighted_histogram = new float[_classCount];
            entropies = new float[_classCount];
            class_prior = new float[_classCount];
            bins = _classCount;

            // Initialize the histogram
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = 0;
                weighted_histogram[i] = 0;
                entropies[i] = 0;
                class_prior[i] = 1;
            }
        }
    }

    /**
     * Returns the size of the histogram (= class count)
     */
    int size() const { return bins; }

    /**
     * Returns the value of the histogram at a certain position. Caution: For performance reasons, we don't
     * perform any parameter check!
     */
    int at(const int i) const { return histogram[i]; }
    int get(const int i) const { return histogram[i]; }
    void set(const int i, const int v) {
      mass -= histogram[i];
      mass += v;
      histogram[i] = v;
      weighted_mass -= weighted_histogram[i];
      weighted_mass += v*class_prior[i];
      weighted_histogram[i] = v*class_prior[i];
    }
    void add(const int i, const int v) {
      mass += v;
      histogram[i] += v;
      weighted_mass += v*class_prior[i];
      weighted_histogram[i] += v*class_prior[i];
    }
    void sub(const int i, const int v) {
      mass -= v;
      histogram[i] -= v;
      weighted_mass -= v*class_prior[i];
      weighted_histogram[i] -= v*class_prior[i];
    }
    void add1(const int i) {
      mass += 1;
      histogram[i]++;
      weighted_mass += class_prior[i];
      weighted_histogram[i] += class_prior[i];
    }
    void sub1(const int i) {
      mass -= 1;
      histogram[i]--;
      weighted_mass -= class_prior[i];
      weighted_histogram[i] -= class_prior[i];
    }
    void addOne(const int i)
    {
        //totalEntropy += ENTROPY(mass);
        totalEntropy += ENTROPY(weighted_mass);
        mass += 1;
        weighted_mass += class_prior[i];
        //totalEntropy -= ENTROPY(mass);
        totalEntropy -= ENTROPY(weighted_mass);
        histogram[i]+=1;
        weighted_histogram[i] += class_prior[i];
        totalEntropy -= entropies[i];
        //entropies[i] = ENTROPY(histogram[i]);
        entropies[i] = ENTROPY(weighted_histogram[i]);
        totalEntropy += entropies[i];
    }
    void subOne(const int i)
    {
        //totalEntropy += ENTROPY(mass);
        totalEntropy += ENTROPY(weighted_mass);
        mass -= 1;
        weighted_mass -= class_prior[i];
        //totalEntropy -= ENTROPY(mass);
        totalEntropy -= ENTROPY(weighted_mass);
        histogram[i]-=1;
        weighted_histogram[i] -= class_prior[i];
        totalEntropy -= entropies[i];
        if (histogram[i] < 1)
        {
            entropies[i] = 0;
        }
        else
        {
            //entropies[i] = ENTROPY(histogram[i]);
            entropies[i] = ENTROPY(weighted_histogram[i]);
            totalEntropy += entropies[i];
        }
    }

    /**
     * Returns the mass
     */
    float getMass() const
    {
        return mass;
    }

    /**
     * Calculates the entropy of a histogram
     *
     * @return The calculated entropy
     */
    float entropy() const
    {
        return totalEntropy;
    }

    /**
     * Initializes all entropies
     */
    void initEntropies()
    {
        if (getMass() > 1)
        {
            totalEntropy = -ENTROPY(getMass());
            for (int i = 0; i < bins; i++)
            {
                if (at(i) == 0) continue;

                entropies[i] = ENTROPY(histogram[i]);

                totalEntropy += entropies[i];
            }
        }
    }

    void initClassPriors(float* c)
    {
      for (int i = 0; i < bins; i++)
      {
        class_prior[i] = c[i];
      }
    }

    /**
     * Sets all entries in the histogram to 0
     */
    void reset()
    {
        for (int i = 0; i < bins; i++)
        {
            histogram[i] = 0;
            weighted_histogram[i] = 0;
            entropies[i] = 0;
            class_prior[i] = 1;
        }
        totalEntropy = 0;
        mass = 0;
        weighted_mass = 0;
    }

    /**
     * Returns the greatest bin
     */
    int argMax() const
    {
        int maxBin = 0;
        int maxCount = histogram[0];
        for (int i = 1; i < bins; i++)
        {
            if (histogram[i] > maxCount)
            {
                maxCount = at(i);
                maxBin = i;
            }
        }

        return maxBin;
    }

    /**
     * Returns true if the histogram is pure
     */
    bool isPure() const
    {
        bool nonPure = false;
        for (int i = 0; i < bins; i++)
        {
            if (histogram[i] > 0)
            {
                if (nonPure)
                {
                    return false;
                }
                else
                {
                    nonPure = true;
                }
            }
        }
        return true;
    }
};

void DecisionTreeLearner::autoconf(const DataStorage* dataStorage)
{
    setUseBootstrap(true);
    setNumBootstrapExamples(dataStorage->getSize());
    setNumFeatures(std::ceil(std::sqrt(dataStorage->getDimensionality())));
}

/**
 * This class can be used in order to sort the array of data point IDs by
 * a certain dimension
 */
class FeatureComparator {
public:
    /**
     * The feature dimension
     */
    int feature;
    /**
     * The data storage
     */
    const DataStorage* storage;

    /**
     * Compares two training examples
     */
    bool operator() (const int lhs, const int rhs) const
    {
        return storage->getDataPoint(lhs)->at(feature) < storage->getDataPoint(rhs)->at(feature);
    }
};

/**
 * Updates the leaf node histograms using a smoothing parameter
 */
inline void updateLeafNodeHistogram(std::vector<float> & leafNodeHistograms, const EfficientEntropyHistogram & hist, float smoothing, bool useBootstrap)
{
    leafNodeHistograms.resize(hist.size());

    if(!useBootstrap)
    {
        for (int c = 0; c < hist.size(); c++)
        {
            leafNodeHistograms[c] = std::log((hist.at(c) + smoothing)/(hist.getMass() + hist.size() * smoothing));
        }
    }
}

DecisionTree* DecisionTreeLearner::learn(const DataStorage* dataStorage) const
{
    if(multipleLabelLayers){
        unsigned int layer_count = dataStorage->getMultiLayerCount();

        DataStorage* storage;
        // If we use bootstrap sampling, then this array contains the results of
        // the sampler. We use it later in order to refine the leaf node histograms
        std::vector<bool> sampled;

        if (useBootstrap)
        {
            storage = new DataStorage(layer_count);
            dataStorage->bootstrapmulti(numBootstrapExamples, storage, sampled);
        }
        else
        {
            storage = new DataStorage(*dataStorage);
        }

        // Get the number of training examples and the dimensionality of the data set
        const int D = storage->getDimensionality();

        // Set up a new tree.
        DecisionTree* tree = new DecisionTree();

        // This is the list of nodes that still have to be split
        std::vector<int> splitStack;
        splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));

        // Add the root node to the list of nodes that still have to be split
        splitStack.push_back(0);

        // This matrix stores the training examples for certain nodes.
        std::vector< int* > trainingExamples;
        std::vector< int > trainingExamplesSizes;
        trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
        trainingExamplesSizes.reserve(LIBF_GRAPH_BUFFER_SIZE);

        // Add all training example to the root node
        trainingExamplesSizes.push_back(storage->getSize());
        trainingExamples.push_back(new int[trainingExamplesSizes[0]]);
        for (int n = 0; n < storage->getSize(); n++)
        {
            trainingExamples[0][n] = n;
        }

        // We keep track on the depth of each node in this array
        // This allows us to stop splitting after a certain depth is reached
        std::vector<int> depths;
        depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
        // The root node has depth 0
        depths.push_back(0);

        // We use this in order to sort the data points
        FeatureComparator cp;
        cp.storage = storage;

        // Set up a probability distribution over the features
        std::mt19937 g(rd());
        // Set up the array of possible features, we use it in order to sample
        // the features without replacement
        std::vector<int> sampledFeatures(D);
        for (int d = 0; d < D; d++)
        {
            sampledFeatures[d] = d;
        }

        // Start training
        while (splitStack.size() > 0)
        {
            // Extract an element from the queue
            const int node = splitStack.back();
            splitStack.pop_back();

            // Get the training example list
            int* trainingExampleList = trainingExamples[node];
            const int N = trainingExamplesSizes[node];

            //Pick a random class layer
            std::uniform_int_distribution<int> dist(0, layer_count -1);
            unsigned int current_layer = dist(rd);

            const int C = storage->getClasscount_multi(current_layer);


            //Count class frequencies.
            std::vector<float> freq(C, 1);
            if(useClassFrequency)
            {
                std::cerr << "Using the class frequency is currently not supported in the multi class case." << std::endl;
            }


            // We use these arrays during training for the left and right histograms
            EfficientEntropyHistogram leftHistogram(C);
            EfficientEntropyHistogram rightHistogram(C);


            // Set up the right histogram
            // Because we start with the threshold being at the left most position
            // The right child node contains all training examples

            EfficientEntropyHistogram hist(C);
            hist.initClassPriors(freq.data());
            for (int m = 0; m < N; m++)
            {
                // Get the class label of this training example
                hist.add1(storage->getClassLabelsMulti(trainingExampleList[m], current_layer));
            }

            // Don't split this node
            //  If the number of examples is too small
            //  If the training examples are all of the same class
            //  If the maximum depth is reached
            if (hist.getMass() < minSplitExamples || hist.isPure() || depths[node] > maxDepth)
            {
                delete[] trainingExampleList;
                // Resize and initialize the leaf node histogram
                //updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter, useBootstrap);
                continue;
            }

            hist.initEntropies();

            // These are the parameters we optimize
            float bestThreshold = 0;
            int bestFeature = -1;
            float bestObjective = 1e35;
            int bestLeftMass = 0;
            int bestRightMass = N;

            // Sample random features
            std::shuffle(sampledFeatures.begin(), sampledFeatures.end(), std::default_random_engine(rd()));

            // Optimize over all features
            for (int f = 0; f < numFeatures; f++)
            {
                const int feature = sampledFeatures[f];

                cp.feature = feature;
                std::sort(trainingExampleList, trainingExampleList + N, cp);

                // Initialize the histograms
                leftHistogram.reset();
                leftHistogram.initClassPriors(freq.data());
                rightHistogram = hist;

                float leftValue = storage->getDataPoint(trainingExampleList[0])->at(feature);
                int leftClass = storage->getClassLabelsMulti(trainingExampleList[0], current_layer);

                // Test different thresholds
                // Go over all examples in this node
                for (int m = 1; m < N; m++)
                {
                    const int n = trainingExampleList[m];

                    // Move the last point to the left histogram
                    leftHistogram.addOne(leftClass);
                    rightHistogram.subOne(leftClass);

                    // It does
                    // Get the two feature values
                    const float rightValue = storage->getDataPoint(n)->at(feature);

                    // Skip this split, if the two points lie too close together
                    const float diff = rightValue - leftValue;

                    if (diff < 1e-6f)
                    {
                        leftValue = rightValue;
                        leftClass = storage->getClassLabelsMulti(n, current_layer);
                        continue;
                    }

                    // Get the objective function
                    const float localObjective = leftHistogram.entropy() + rightHistogram.entropy();

                    if (localObjective < bestObjective)
                    {
                        // Get the threshold value
                        bestThreshold = (leftValue + rightValue);
                        bestFeature = feature;
                        bestObjective = localObjective;
                        bestLeftMass = leftHistogram.getMass();
                        bestRightMass = rightHistogram.getMass();
                    }

                    leftValue = rightValue;
                    leftClass = storage->getClassLabelsMulti(n, current_layer);
                }
            }
            // We spare the additional multiplication at each iteration.
            bestThreshold *= 0.5f;

            // Did we find good split values?
            if (bestFeature < 0 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
            {
                // We didn't
                // Don't split
                delete[] trainingExampleList;
                //updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter, useBootstrap);
                continue;
            }

            // Set up the data lists for the child nodes
            trainingExamplesSizes.push_back(bestLeftMass);
            trainingExamplesSizes.push_back(bestRightMass);
            trainingExamples.push_back(new int[bestLeftMass]);
            trainingExamples.push_back(new int[bestRightMass]);

            int* leftList = trainingExamples[trainingExamples.size() - 2];
            int* rightList = trainingExamples[trainingExamples.size() - 1];

            // Sort the points
            for (int m = 0; m < N; m++)
            {
                const int n = trainingExampleList[m];
                const float featureValue = storage->getDataPoint(n)->at(bestFeature);

                if (featureValue < bestThreshold)
                {
                    leftList[--bestLeftMass] = n;
                }
                else
                {
                    rightList[--bestRightMass] = n;
                }
            }

            // Ok, split the node
            tree->setThreshold(node, bestThreshold);
            tree->setSplitFeature(node, bestFeature);
            const int leftChild = tree->splitNode(node);

            // Update the depth
            depths.push_back(depths[node] + 1);
            depths.push_back(depths[node] + 1);

            // Prepare to split the child nodes
            splitStack.push_back(leftChild);
            splitStack.push_back(leftChild + 1);

            delete[] trainingExampleList;
        }

        // Free the data set
        delete storage;
        std::cout << "almost done!" << std::endl;
        // Use all the training examples for the multi histograms.
        updateMultiHistograms(tree, dataStorage);

        return tree;
    }else{
        DataStorage* storage;
        // If we use bootstrap sampling, then this array contains the results of
        // the sampler. We use it later in order to refine the leaf node histograms
        std::vector<bool> sampled;

        if (useBootstrap)
        {
            storage = new DataStorage;
            dataStorage->bootstrap(numBootstrapExamples, storage, sampled);
        }
        else
        {
            storage = new DataStorage(*dataStorage);
        }

        // Get the number of training examples and the dimensionality of the data set
        const int D = storage->getDimensionality();
        const int C = storage->getClasscount();

        // Set up a new tree.
        DecisionTree* tree = new DecisionTree();

        // This is the list of nodes that still have to be split
        std::vector<int> splitStack;
        splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));

        // Add the root node to the list of nodes that still have to be split
        splitStack.push_back(0);

        // This matrix stores the training examples for certain nodes.
        std::vector< int* > trainingExamples;
        std::vector< int > trainingExamplesSizes;
        trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
        trainingExamplesSizes.reserve(LIBF_GRAPH_BUFFER_SIZE);

        // Add all training example to the root node
        trainingExamplesSizes.push_back(storage->getSize());
        trainingExamples.push_back(new int[trainingExamplesSizes[0]]);
        for (int n = 0; n < storage->getSize(); n++)
        {
            trainingExamples[0][n] = n;
        }


        //Count class frequencies.
        std::vector<float> freq;
        if(useClassFrequency)
        {
            freq = storage->getInvertedClassFrequency();
        }
        else
        {
            freq = std::vector<float>(storage->getClasscount(), 1);
        }


        // We use these arrays during training for the left and right histograms
        EfficientEntropyHistogram leftHistogram(C);
        EfficientEntropyHistogram rightHistogram(C);


        // We keep track on the depth of each node in this array
        // This allows us to stop splitting after a certain depth is reached
        std::vector<int> depths;
        depths.reserve(LIBF_GRAPH_BUFFER_SIZE);
        // The root node has depth 0
        depths.push_back(0);

        // We use this in order to sort the data points
        FeatureComparator cp;
        cp.storage = storage;

        // Set up a probability distribution over the features
        std::mt19937 g(rd());
        // Set up the array of possible features, we use it in order to sample
        // the features without replacement
        std::vector<int> sampledFeatures(D);
        for (int d = 0; d < D; d++)
        {
            sampledFeatures[d] = d;
        }

        // Start training
        while (splitStack.size() > 0)
        {
            // Extract an element from the queue
            const int node = splitStack.back();
            splitStack.pop_back();

            // Get the training example list
            int* trainingExampleList = trainingExamples[node];
            const int N = trainingExamplesSizes[node];

            // Set up the right histogram
            // Because we start with the threshold being at the left most position
            // The right child node contains all training examples

            EfficientEntropyHistogram hist(C);
            hist.initClassPriors(freq.data());
            for (int m = 0; m < N; m++)
            {
                // Get the class label of this training example
                hist.add1(storage->getClassLabel(trainingExampleList[m]));
            }

            // Don't split this node
            //  If the number of examples is too small
            //  If the training examples are all of the same class
            //  If the maximum depth is reached
            if (hist.getMass() < minSplitExamples || hist.isPure() || depths[node] > maxDepth)
            {
                delete[] trainingExampleList;
                // Resize and initialize the leaf node histogram
                updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter, useBootstrap);
                continue;
            }

            hist.initEntropies();

            // These are the parameters we optimize
            float bestThreshold = 0;
            int bestFeature = -1;
            float bestObjective = 1e35;
            int bestLeftMass = 0;
            int bestRightMass = N;

            // Sample random features
            std::shuffle(sampledFeatures.begin(), sampledFeatures.end(), std::default_random_engine(rd()));

            // Optimize over all features
            for (int f = 0; f < numFeatures; f++)
            {
                const int feature = sampledFeatures[f];

                cp.feature = feature;
                std::sort(trainingExampleList, trainingExampleList + N, cp);

                // Initialize the histograms
                leftHistogram.reset();
                leftHistogram.initClassPriors(freq.data());
                rightHistogram = hist;

                float leftValue = storage->getDataPoint(trainingExampleList[0])->at(feature);
                int leftClass = storage->getClassLabel(trainingExampleList[0]);

                // Test different thresholds
                // Go over all examples in this node
                for (int m = 1; m < N; m++)
                {
                    const int n = trainingExampleList[m];

                    // Move the last point to the left histogram
                    leftHistogram.addOne(leftClass);
                    rightHistogram.subOne(leftClass);

                    // It does
                    // Get the two feature values
                    const float rightValue = storage->getDataPoint(n)->at(feature);

                    // Skip this split, if the two points lie too close together
                    const float diff = rightValue - leftValue;

                    if (diff < 1e-6f)
                    {
                        leftValue = rightValue;
                        leftClass = storage->getClassLabel(n);
                        continue;
                    }

                    // Get the objective function
                    const float localObjective = leftHistogram.entropy() + rightHistogram.entropy();

                    if (localObjective < bestObjective)
                    {
                        // Get the threshold value
                        bestThreshold = (leftValue + rightValue);
                        bestFeature = feature;
                        bestObjective = localObjective;
                        bestLeftMass = leftHistogram.getMass();
                        bestRightMass = rightHistogram.getMass();
                    }

                    leftValue = rightValue;
                    leftClass = storage->getClassLabel(n);
                }
            }
            // We spare the additional multiplication at each iteration.
            bestThreshold *= 0.5f;

            // Did we find good split values?
            if (bestFeature < 0 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
            {
                // We didn't
                // Don't split
                delete[] trainingExampleList;
                updateLeafNodeHistogram(tree->getHistogram(node), hist, smoothingParameter, useBootstrap);
                continue;
            }

            // Set up the data lists for the child nodes
            trainingExamplesSizes.push_back(bestLeftMass);
            trainingExamplesSizes.push_back(bestRightMass);
            trainingExamples.push_back(new int[bestLeftMass]);
            trainingExamples.push_back(new int[bestRightMass]);

            int* leftList = trainingExamples[trainingExamples.size() - 2];
            int* rightList = trainingExamples[trainingExamples.size() - 1];

            // Sort the points
            for (int m = 0; m < N; m++)
            {
                const int n = trainingExampleList[m];
                const float featureValue = storage->getDataPoint(n)->at(bestFeature);

                if (featureValue < bestThreshold)
                {
                    leftList[--bestLeftMass] = n;
                }
                else
                {
                    rightList[--bestRightMass] = n;
                }
            }

            // Ok, split the node
            tree->setThreshold(node, bestThreshold);
            tree->setSplitFeature(node, bestFeature);
            const int leftChild = tree->splitNode(node);

            // Update the depth
            depths.push_back(depths[node] + 1);
            depths.push_back(depths[node] + 1);

            // Prepare to split the child nodes
            splitStack.push_back(leftChild);
            splitStack.push_back(leftChild + 1);

            delete[] trainingExampleList;
        }

        // Free the data set
        delete storage;

        // If we use bootstrap, we use all the training examples for the
        // histograms
        if (useBootstrap)
        {
            updateHistograms(tree, dataStorage);
        }

        return tree;
    }
}

void DecisionTreeLearner::updateHistograms(DecisionTree* tree, const DataStorage* storage) const
{
    const int C = storage->getClasscount();

    // Reset all histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->isLeafNode(v))
        {
            std::vector<float> & hist = tree->getHistogram(v);
            for (int c = 0; c < C; c++)
            {
                hist[c] = 0;
            }
        }
    }

    std::vector<float> freq = storage->getInvertedClassFrequency();
    // Compute the weights for each data point
    for (int n = 0; n < storage->getSize(); n++)
    {
        int leafNode = tree->findLeafNode(storage->getDataPoint(n));
        auto classlabel = storage->getClassLabel(n);
        tree->getHistogram(leafNode)[classlabel] += freq[classlabel];
    }

    // Normalize the histograms
    for (int v = 0; v < tree->getNumNodes(); v++)
    {
        if (tree->isLeafNode(v))
        {
            std::vector<float> & hist = tree->getHistogram(v);
            float total = 0;
            for (int c = 0; c < C; c++)
            {
                total += hist[c];
            }
            for (int c = 0; c < C; c++)
            {
                hist[c] = std::log((hist[c] + smoothingParameter)/(total + C*smoothingParameter));
            }
        }
    }
}

void DecisionTreeLearner::updateMultiHistograms(DecisionTree* tree, const DataStorage* storage) const
{
    unsigned int layer_count = storage->getMultiLayerCount();
    for(unsigned int l=0; l < layer_count; l++){
        const int C = storage->getClasscount_multi(l);
        std::cout << C << std::endl;

        // Reset all histograms
        for (int v = 0; v < tree->getNumNodes(); v++)
        {
            if (tree->isLeafNode(v))
            {
                std::vector<std::vector<float> > & hist = tree->getMultiHistogram(v);
                //If this is the first time we touch the node, we will have to create the vector of hists first.
                if(l == 0){
                    hist = std::vector<std::vector<float> > (layer_count);
                }
                hist[l] = std::vector<float>(C, 0);
            }
        }

        std::vector<float> freq = storage->getInvertedClassFrequency(l);

        // Compute the weights for each data point
        for (int n = 0; n < storage->getSize(); n++)
        {
            int leafNode = tree->findLeafNode(storage->getDataPoint(n));
            auto classlabel = storage->getClassLabelsMulti(n, l);
            tree->getMultiHistogram(leafNode)[l][classlabel] += freq[classlabel];
        }

        // Normalize the histograms
        for (int v = 0; v < tree->getNumNodes(); v++)
        {
            if (tree->isLeafNode(v))
            {
                std::vector<std::vector<float> > & hist = tree->getMultiHistogram(v);
                float total = 0;
                for (int c = 0; c < C; c++)
                {
                    total += hist[l][c];
                }
                for (int c = 0; c < C; c++)
                {
                    hist[l][c] = std::log((hist[l][c] + smoothingParameter)/(total + C*smoothingParameter));
                }
            }
        }
    }
}


void DecisionTreeLearner::dumpSetting(std::ostream & stream) const
{
    stream << std::setw(30) << "Learner" << ": DecisionTreeLearner" << "\n";
    stream << std::setw(30) << "Bootstrap Sampling" << ": " << getUseBootstrap() << "\n";
    stream << std::setw(30) << "Bootstrap Samples" << ": " << getNumBootstrapExamples() << "\n";
    stream << std::setw(30) << "Feature evaluations" << ": " << getNumFeatures() << "\n";
    stream << std::setw(30) << "Max depth" << ": " << getMaxDepth() << "\n";
    stream << std::setw(30) << "Minimum Split Examples" << ": " << getMinSplitExamples() << "\n";
    stream << std::setw(30) << "Minimum Child Split Examples" << ": " << getMinChildSplitExamples() << "\n";
    stream << std::setw(30) << "Smoothing Parameter" << ": " << getSmoothingParameter() << "\n";
}

////////////////////////////////////////////////////////////////////////////////
/// RandomForestLearner
////////////////////////////////////////////////////////////////////////////////

RandomForest* RandomForestLearner::learn(const DataStorage* storage) const
{
    // Set up the empty random forest
    RandomForest* forest = new RandomForest();

    // Set up the state for the call backs
    RandomForestLearnerState state;
    state.learner = this;
    state.forest = forest;
    state.action = ACTION_START_FOREST;

    evokeCallback(forest, 0, &state);

    int treeStartCounter = 0;
    int treeFinishCounter = 0;
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < numTrees; i++)
    {
        #pragma omp critical
        {
            state.tree = ++treeStartCounter;
            state.action = ACTION_START_TREE;
            evokeCallback(forest, treeStartCounter - 1, &state);
        }

        // Learn the tree
        DecisionTree* tree = treeLearner->learn(storage);
        // Add it to the forest
        #pragma omp critical
        {
            state.tree = ++treeFinishCounter;
            state.action = ACTION_FINISH_TREE;
            evokeCallback(forest, treeFinishCounter - 1, &state);
            forest->addTree(tree);
        }
    }

    state.tree = 0;
    state.action = ACTION_FINISH_FOREST;
    evokeCallback(forest, 0, &state);

    return forest;
}

void RandomForestLearner::dumpSetting(std::ostream& stream) const
{
    stream << std::setw(30) << "Learner" << ": RandomForestLearner" << "\n";
    stream << std::setw(30) << "Number of trees" << ": " << getNumTrees() << "\n";
    stream << std::setw(30) << "Number of threads" << ": " << getNumThreads() << "\n";
    stream << "Tree learner settings" << "\n";
    treeLearner->dumpSetting(stream);
}

int RandomForestLearner::defaultCallback(RandomForest* forest, RandomForestLearnerState* state)
{
    switch (state->action) {
        case RandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start random forest training\n";
            state->learner->dumpSetting();
            std::cout << "\n";
            break;
        case RandomForestLearner::ACTION_START_TREE:
            std::cout   << std::setw(15) << std::left << "Start tree "
                        << std::setw(4) << std::right << state->tree
                        << " out of "
                        << std::setw(4) << state->learner->getNumTrees() << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_TREE:
            std::cout   << std::setw(15) << std::left << "Finish tree "
                        << std::setw(4) << std::right << state->tree
                        << " out of "
                        << std::setw(4) << state->learner->getNumTrees() << "\n";
            break;
        case RandomForestLearner::ACTION_FINISH_FOREST:
            std::cout << "Finished forest in " << state->getPassedTime().count()/1000000. << "s\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state->action << "\n";
            break;
    }
    return 0;

}


////////////////////////////////////////////////////////////////////////////////
/// BoostedRandomForestLearner
////////////////////////////////////////////////////////////////////////////////

BoostedRandomForest* BoostedRandomForestLearner::learn(const DataStorage* storage) const
{
    // Set up the empty random forest
    BoostedRandomForest* forest = new BoostedRandomForest();

    // Set up the state for the call backs
    BoostedRandomForestLearnerState state;
    state.learner = this;
    state.forest = forest;
    state.action = ACTION_START_FOREST;

    evokeCallback(forest, 0, &state);

    // Set up the weights for the data points
    const int N = storage->getSize();
    std::vector<float> dataWeights(N);
    std::vector<float> cumsum(N);
    std::vector<bool> misclassified(N);
    for (int n = 0; n < N; n++)
    {
        dataWeights[n] = 1.0f/N;
        cumsum[n] = (n+1) * 1.0f/N;
        misclassified[n] = false;
    }

    // We need this distribution in order to sample according to the weights
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> U(0, 1);

    const int C = storage->getClasscount();

    int treeStartCounter = 0;
    int treeFinishCounter = 0;
    for (int i = 0; i < numTrees; i++)
    {
        state.tree = ++treeStartCounter;
        state.action = ACTION_START_TREE;
        evokeCallback(forest, treeStartCounter - 1, &state);

        // Learn the tree
        // --------------

        // Sample data points according to the weights
        DataStorage treeData;
        treeData.setClasscount(storage->getClasscount());

        for (int n = 0; n < N; n++)
        {
            const float u = U(g);
            int index = 0;
            while (u > cumsum[index])
            {
                index++;
            }
            treeData.addDataPoint(storage->getDataPoint(index), storage->getClassLabel(index), false);
        }

        // Learn the tree
        DecisionTree* tree = treeLearner->learn(&treeData);

        // Calculate the error term
        float error = 0;
        for (int n = 0; n < N; n++)
        {
            const int predictedLabel = tree->classify(storage->getDataPoint(n));
            if (predictedLabel != storage->getClassLabel(n))
            {
                error += dataWeights[n];
                misclassified[n] = true;
            }
            else
            {
                misclassified[n] = false;
            }
        }

        // Compute the classifier weight
        const float alpha = std::log((1-error)/error) + std::log(C - 1);

        std::cout << "error = " << error << ", alpha = " << alpha << "\n";

        // Update the weights
        float total = 0;
        for (int n = 0; n < N; n++)
        {
            if (misclassified[n])
            {
                dataWeights[n] *= std::exp(alpha);
            }
            total += dataWeights[n];
        }
        dataWeights[0] /= total;
        cumsum[0] = dataWeights[0];
        for (int n = 1; n < N; n++)
        {
            dataWeights[n] /= total;
            cumsum[n] = dataWeights[n] + cumsum[n-1];
        }

        // Add the classifier
        forest->addTree(tree, alpha);

        // --------------
        // Add it to the forest
        state.tree = ++treeFinishCounter;
        state.action = ACTION_FINISH_TREE;
        evokeCallback(forest, treeFinishCounter - 1, &state);
    }

    state.tree = 0;
    state.action = ACTION_FINISH_FOREST;
    evokeCallback(forest, 0, &state);

    return forest;
}


void BoostedRandomForestLearner::dumpSetting(std::ostream& stream) const
{
    stream << std::setw(30) << "Learner" << ": BoostedRandomForestLearner" << "\n";
    stream << std::setw(30) << "Number of trees" << ": " << getNumTrees() << "\n";
    stream << "Tree learner settings" << "\n";
    treeLearner->dumpSetting(stream);
}

int BoostedRandomForestLearner::defaultCallback(BoostedRandomForest* forest, BoostedRandomForestLearnerState* state)
{
    switch (state->action) {
        case BoostedRandomForestLearner::ACTION_START_FOREST:
            std::cout << "Start boosted random forest training\n";
            state->learner->dumpSetting();
            std::cout << "\n";
            break;
        case BoostedRandomForestLearner::ACTION_START_TREE:
            std::cout   << std::setw(15) << std::left << "Start tree "
                        << std::setw(4) << std::right << state->tree
                        << " out of "
                        << std::setw(4) << state->learner->getNumTrees() << "\n";
            break;
        case BoostedRandomForestLearner::ACTION_FINISH_TREE:
            std::cout   << std::setw(15) << std::left << "Finish tree "
                        << std::setw(4) << std::right << state->tree
                        << " out of "
                        << std::setw(4) << state->learner->getNumTrees() << "\n";
            break;
        case BoostedRandomForestLearner::ACTION_FINISH_FOREST:
            std::cout << "Finished boosted forest in " << state->getPassedTime().count()/1000000. << "s\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state->action << "\n";
            break;
    }
    return 0;
}
