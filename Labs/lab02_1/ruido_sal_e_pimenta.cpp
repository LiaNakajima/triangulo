void addSaltAndPepper(Mat& image, double salt_prob, double pepper_prob) {
    RNG rng; // Random number generator
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            double random_value = rng.uniform(0.0, 1.0);
            if (random_value < pepper_prob) {
                image.at<Vec3b>(i, j) = Vec3b(0, 0, 0); // Pimenta (preto)
            } else if (random_value > 1.0 - salt_prob) {
                image.at<Vec3b>(i, j) = Vec3b(255, 255, 255); // Sal (branco)
            }
        }
    }
}
